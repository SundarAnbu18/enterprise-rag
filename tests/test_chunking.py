"""The chunker: paragraphs in, retrievable units out."""

from django.test import SimpleTestCase

from ragengine.chunking import MAX_CHUNK_CHARS, Chunk, chunk_document, split_paragraphs


class SplitParagraphsTests(SimpleTestCase):
    def test_splits_on_blank_lines(self):
        text = "First paragraph.\n\nSecond paragraph.\n\n\nThird."
        self.assertEqual(
            split_paragraphs(text), ["First paragraph.", "Second paragraph.", "Third."]
        )

    def test_collapses_internal_whitespace(self):
        self.assertEqual(split_paragraphs("a  line\nwrapped   here"), ["a line wrapped here"])

    def test_empty_text_gives_nothing(self):
        self.assertEqual(split_paragraphs("   \n\n  "), [])


class ChunkDocumentTests(SimpleTestCase):
    def test_tags_every_chunk_with_source(self):
        chunks = chunk_document("One.\n\nTwo.", source="faq.md")
        self.assertEqual([c.source for c in chunks], ["faq.md", "faq.md"])
        self.assertEqual([c.text for c in chunks], ["One.", "Two."])

    def test_long_paragraph_is_split_on_sentences(self):
        sentence = "This sentence pads the paragraph out to a serious length. "
        text = sentence * 40  # far beyond MAX_CHUNK_CHARS, no blank lines
        chunks = chunk_document(text, source="long.md")
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk.text), MAX_CHUNK_CHARS)

    def test_dict_roundtrip(self):
        chunk = Chunk(text="hello", source="a.md")
        self.assertEqual(Chunk.from_dict(chunk.to_dict()), chunk)
