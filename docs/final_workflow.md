# Implemented Deed OCR Workflows

## Overview

This document outlines the two primary, implemented workflows for deed OCR processing within this project. Both workflows leverage Google's Gemini AI models but differ in their architecture and intermediate steps. This document serves as an up-to-date reference for the code in `deed_ocr/workflow.py` and `deed_ocr/workflow_vision.py`, reflecting the "as-built" system, in contrast to the conceptual `llm-enhanced-architecture.md`.

The two workflows are:
1.  **Simplified Gemini OCR Workflow**: A multi-stage Gemini-only workflow that performs both per-page and full-document analysis and then merges the results.
2.  **Vision + Gemini Workflow**: A two-stage pipeline that uses Google Cloud Vision for initial text extraction (OCR) and then Gemini AI for structuring the extracted text.

---

## Workflow 1: Simplified Gemini OCR Workflow

This workflow is implemented in `deed_ocr/workflow.py` and uses Gemini AI for all OCR and structuring tasks. It is designed as a comprehensive process that analyzes documents at both page and full-document levels to maximize data extraction accuracy.

### Architecture Diagram

1.  **PDF Input**
    -   Starts with the source PDF file.
    -   Branches into two parallel processes:
        1.  **PDF to Images -> Per-Page Analysis**
        2.  **Full Document Analysis**

2.  **Per-Page Analysis**
    -   **PDF to Images**: The PDF is split into individual page images (Page 1, Page 2, ... Page N).
    -   **Gemini AI (Simplified Prompt)**: Each page image is sent to Gemini for initial, fast extraction.
    -   **Output**: This produces per-page JSON files containing `full_text`, `legal_desc`, `TRS`, etc.

3.  **Full Document Analysis**
    -   **Gemini AI (Full PDF Prompt)**: The entire original PDF is sent to Gemini for a holistic analysis.
    -   **Output**: This produces a single, comprehensive JSON object with `doc_type`, `parties`, and detailed `deed`/`lease` information.

4.  **Merge & Finalize**
    -   **Merge & Post-Process**: The results from both the per-page analysis and the full document analysis are combined. Duplicates are removed and data is cleaned.
    -   **Outputs**:
        -   **Final Structured JSON Output (`final_result.json`)**: The primary, merged result file.
        -   **Other Outputs**: Includes `full_text.txt`, `token_usage.json`, etc.

### Detailed Workflow Steps

1.  **PDF to Image Conversion**: The input PDF is converted into a series of high-resolution PNG images, one for each page.
2.  **Per-Page Simplified Extraction**: Each page image is processed individually by Gemini AI using a *simplified prompt* (`process_image_bytes_simplified`). This step is designed for fast and targeted extraction of key elements from each page.
    -   **Prompt Goal**: Extract `full_text`, `legal_description_block`, `reserve_retain` sentences, `oil_mineral` sentences, and `TRS` references.
    -   **Output**: A JSON object for each page containing the extracted data.
3.  **Full Document Analysis**: The entire, original PDF file is processed in a single call to Gemini AI using a more *detailed, multi-page prompt* (`process_entire_pdf`). This leverages the model's ability to understand context across the whole document.
    -   **Prompt Goal**: Extract high-level document details like `document_type`, `parties`, `deed_details`, `lease_details`, and cross-page `legal_description_block` and `TRS` references.
    -   **Output**: A single JSON object with the comprehensive analysis.
4.  **Result Merging (`_create_final_result`)**: The data from the per-page analysis and the full-document analysis are merged into a single, cohesive result.
    -   The `full_pdf_analysis.json` serves as the base.
    -   `full_text` is aggregated from all pages.
    -   `legal_description_block` and `TRS` lists are merged from both sources, with duplicates removed.
    -   `reserve_retain` and `oil_mineral` sentences are taken from the per-page analysis.
5.  **Post-Processing (`_post_process_results`)**: The merged result undergoes a final cleanup.
    -   Removes common OCR watermarks (e.g., "UNOFFICIAL COPY") from the full text.
    -   Removes duplicate string entries from all lists within the final JSON object.
6.  **Saving Outputs**: The workflow saves multiple files into a directory named after the source PDF. The primary output is `final_result.json`.

### Final Output Structure (`final_result.json`)

The `final_result.json` file contains the complete, merged, and cleaned data.

```json
{
  "legal_description_block": [
    "<Merged and deduplicated legal property descriptions from all sources>"
  ],
  "details": {
    "document_type": "<Primary document category>",
    "document_subtype": "<Specific document type>",
    "parties": {
      "grantor": ["<Grantor names>"],
      "grantee": ["<Grantee names>"]
    },
    "deed_details": {
      "grantors_interest": "<grantors interest>",
      "Interest_fraction": "<interest transferred>",
      "subject_to": "<subject to clauses>"
    },
    "lease_details": {
      "gross_acreage": "<acreage>",
      "lease_royalty": "<royalty>",
      "lease_term": "<term>"
    }
  },
  "token_usage": {
    "input_tokens": 0,
    "output_tokens": 0,
    "total_tokens": 0
  },
  "full_text": "<Cleaned, combined full text from all pages>",
  "TRS": ["<Merged and deduplicated Township/Range/Section references>"],
  "reserve_retain": ["<Sentences containing reservation/exception clauses>"],
  "oil_mineral": ["<Sentences mentioning 'oil' or 'minerals'>"]
}
```

---

## Workflow 2: Vision + Gemini Workflow

This workflow is implemented in `deed_ocr/workflow_vision.py`. It separates the OCR task from the data structuring task, using Google's specialized Cloud Vision API for text extraction before sending the text to Gemini for analysis.

### Architecture Diagram

1.  **PDF Input**
    -   Starts with the source PDF file.

2.  **Text Extraction (OCR)**
    -   **PDF to Images**: The PDF is converted into individual page images.
    -   **Google Cloud Vision API**: The images are sent to the Vision API for OCR.
    -   **Output**: Produces raw text for each page.
        -   *Side Output*: Saves the raw `vision_results.json`.

3.  **Combine and Structure**
    -   **Combine Text**: Text from all pages is concatenated into one large string.
    -   **Gemini AI (Structure from Text Prompt)**: The combined text is sent to Gemini AI for structuring.
    -   **Output**: Produces a single structured JSON from Gemini's analysis.

4.  **Finalization**
    -   **Post-Process**: The Gemini JSON output and the combined text are cleaned (e.g., watermarks and duplicates are removed).
    -   **Final Structured JSON Output (`complete_result.json`)**: The final, cleaned, and comprehensive JSON result is created and saved.

### Detailed Workflow Steps

1.  **PDF to Image Conversion**: The input PDF is converted into a series of high-resolution PNG images, one per page.
2.  **Text Extraction (OCR)**: All page images are sent to the Google Cloud Vision API (`extract_text_from_multiple_images`), which performs OCR and returns the extracted text for each page along with coordinates and confidence scores. The raw Vision API output is saved in `vision_results.json`.
3.  **Combine Text**: The raw text from all pages is concatenated into a single string.
4.  **Text Structuring**: The combined text string is sent to Gemini AI with a prompt (`process_extracted_text`) instructing it to structure the provided text into a JSON format.
    -   **Prompt Goal**: Similar to the "Full Document Analysis" in Workflow 1, this prompt extracts `legal_description_block`, `reserve_retain`, `oil_mineral`, and a `details` object containing `document_type`, `parties`, `TRS`, `deed_details`, and `lease_details`.
    -   **Output**: A single structured JSON object. This is saved as `gemini_structured.json`.
5.  **Post-Processing (`_post_process_results`)**: The text and JSON from Gemini undergo a cleanup process.
    -   Removes common OCR watermarks from the full text.
    -   Removes duplicate string entries from lists within the Gemini JSON output.
6.  **Saving Outputs**: The workflow saves all artifacts into a directory. The main file, `complete_result.json`, bundles all information together.

### Final Output Structure (`complete_result.json`)

This file is a serialization of the `VisionGeminiDeedResult` Pydantic model and contains the results from all stages.

```json
{
  "source_pdf": "<filename.pdf>",
  "total_pages": 0,
  "vision_results": {
    "combined_text": "<Raw text from Vision, with watermarks>",
    "total_pages": 0,
    "page_results": [
      {
        "page_number": 1,
        "text": "<Raw text for page 1>",
        "character_count": 0
      }
    ]
  },
  "gemini_structured_result": {
    "legal_description_block": ["<Deduplicated legal descriptions>"],
    "reserve_retain": ["<Deduplicated reservation/exception clauses>"],
    "oil_mineral": ["<Deduplicated 'oil'/'minerals' sentences>"],
    "details": {
      "document_type": "<Primary document category>",
      "document_subtype": "<Specific document type>",
      "parties": {
        "grantor": ["<Grantor names>"],
        "grantee": ["<Grantee names>"]
      },
      "TRS": ["<Township/Range/Section references>"],
      "deed_details": {
        "grantors_interest": null,
        "Interest_fraction": null,
        "subject_to": null
      },
      "lease_details": {
        "gross_acreage": null,
        "lease_royalty": null,
        "lease_term": null
      }
    },
    "token_usage": {
      "input_tokens": 0,
      "output_tokens": 0,
      "total_tokens": 0
    }
  },
  "combined_full_text": "<Cleaned full text from Vision OCR>",
  "has_errors": false,
  "error_summary": {},
  "retry_needed": false
}
``` 