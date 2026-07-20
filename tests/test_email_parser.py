"""
Unit tests for email_parser.parse_eml's filename field — added so
digest/core/citations.py has a real, openable file reference for each
email (message_id alone isn't something a reviewer can directly open).

Run: python3 -m unittest test_email_parser -v
"""

import os
import tempfile
import unittest

from digest.parsers.email_parser import parse_eml

_SAMPLE_EML = (
    "Subject: Test subject\n"
    "From: Alice <alice@example.com>\n"
    "To: Bob <bob@example.com>\n"
    "Date: Wed, 01 Jul 2026 09:00:00 -0500\n"
    "Message-ID: <test-1@example.com>\n"
    "\n"
    "Hello, this is a test body.\n"
)


class TestParseEmlFilename(unittest.TestCase):
    def test_filename_is_basename_of_filepath(self):
        with tempfile.TemporaryDirectory() as tmp:
            filepath = os.path.join(tmp, "0007.eml")
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(_SAMPLE_EML)
            email = parse_eml(filepath)
            self.assertEqual(email["filename"], "0007.eml")

    def test_filename_ignores_directory_components(self):
        with tempfile.TemporaryDirectory() as tmp:
            nested = os.path.join(tmp, "nested", "dir")
            os.makedirs(nested)
            filepath = os.path.join(nested, "0012.eml")
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(_SAMPLE_EML)
            email = parse_eml(filepath)
            self.assertEqual(email["filename"], "0012.eml")

    def test_other_fields_still_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            filepath = os.path.join(tmp, "0001.eml")
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(_SAMPLE_EML)
            email = parse_eml(filepath)
            self.assertEqual(email["subject"], "Test subject")
            self.assertEqual(email["message_id"], "<test-1@example.com>")


if __name__ == "__main__":
    unittest.main()
