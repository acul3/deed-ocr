"""Tests for legal description extraction."""

import pytest

from deed_ocr.main import LegalDescription, PageText, find_legal_descriptions


class TestLegalDescriptionExtraction:
    """Test legal description extraction functionality."""

    @pytest.fixture
    def sample_page_with_legal_description(self):
        """Sample page text containing a legal description."""
        return PageText(
            page_number=1,
            text="""
            DEED OF TRUST
            
            This deed made this 15th day of March, 2023...
            
            LEGAL DESCRIPTION:
            
            Lot 5, Block 3, of SUNNY ACRES SUBDIVISION, according to the plat
            thereof recorded in Plat Book 123, Page 45, of the Public Records
            of Example County, State.
            
            TOGETHER WITH all improvements thereon...
            """,
            confidence=0.98,
        )

    @pytest.fixture
    def sample_page_without_legal_description(self):
        """Sample page text without a legal description."""
        return PageText(
            page_number=2,
            text="""
            EXHIBIT A
            
            Terms and conditions of the mortgage...
            
            1. Payment shall be made monthly
            2. Interest rate is fixed at 5.5%
            """,
            confidence=0.97,
        )

    def test_find_legal_description_in_text(self, sample_page_with_legal_description):
        """Test finding legal description in page with clear legal description."""
        pages = [sample_page_with_legal_description]
        results = find_legal_descriptions(pages)
        
        # Should find at least one legal description
        assert len(results) > 0
        assert isinstance(results[0], LegalDescription)
        assert results[0].page_number == 1

    def test_no_legal_description_found(self, sample_page_without_legal_description):
        """Test handling of pages without legal descriptions."""
        pages = [sample_page_without_legal_description]
        results = find_legal_descriptions(pages)
        
        # Should not find any legal descriptions
        assert len(results) == 0

    def test_multiple_pages_processing(
        self,
        sample_page_with_legal_description,
        sample_page_without_legal_description,
    ):
        """Test processing multiple pages."""
        pages = [
            sample_page_with_legal_description,
            sample_page_without_legal_description,
        ]
        results = find_legal_descriptions(pages)
        
        # Should find legal description only on first page
        assert len(results) >= 1
        assert all(desc.page_number == 1 for desc in results)


@pytest.mark.parametrize(
    "text,expected_found",
    [
        ("Beginning at a point on the North line of Section 12...", True),
        ("Lot 42, Block B, RIVERSIDE ESTATES", True),
        ("The SW 1/4 of Section 15, Township 2N, Range 3E", True),
        ("This is just a regular paragraph with no legal description", False),
    ],
)
def test_legal_description_patterns(text, expected_found):
    """Test various legal description patterns."""
    page = PageText(page_number=1, text=text, confidence=0.95)
    results = find_legal_descriptions([page])
    
    if expected_found:
        assert len(results) > 0, f"Should find legal description in: {text}"
    else:
        assert len(results) == 0, f"Should not find legal description in: {text}" 