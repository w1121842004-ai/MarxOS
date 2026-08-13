from __future__ import annotations

import hashlib
import json
import unittest

from marxos.indexing.milvus_contract_v2 import (
    BASE_OUTPUT_FIELDS,
    CORE_FIELD_NAMES,
    CONTRACT_VERSION,
    OPTIONAL_SPARSE_FIELD,
    canonical_record_id,
    create_v2_schema,
    from_milvus_int,
    row_from_record_v2,
    text_sha256,
    to_milvus_int,
)


class _DataType:
    VARCHAR = "varchar"
    INT64 = "int64"
    BOOL = "bool"
    FLOAT_VECTOR = "float_vector"
    SPARSE_FLOAT_VECTOR = "sparse_float_vector"


class _Schema:
    def __init__(self):
        self.fields = []

    def add_field(self, name, data_type, **kwargs):
        self.fields.append((name, data_type, kwargs))


class _Indexes:
    def __init__(self):
        self.indexes = []

    def add_index(self, **kwargs):
        self.indexes.append(kwargs)


class _Client:
    def __init__(self, existing=False):
        self.existing = existing
        self.dropped = []
        self.schema = None
        self.indexes = None

    def has_collection(self, name):
        return self.existing

    def drop_collection(self, name):
        self.dropped.append(name)
        self.existing = False

    def create_schema(self, **_kwargs):
        self.schema = _Schema()
        return self.schema

    def prepare_index_params(self):
        self.indexes = _Indexes()
        return self.indexes

    def create_collection(self, collection_name, schema, index_params):
        self.created = (collection_name, schema, index_params)


class MilvusContractV2Tests(unittest.TestCase):
    def sample_record(self):
        return {
            "source": "mea04.pdf",
            "source_file": "马克思恩格斯全集-04.pdf",
            "book": "马克思恩格斯全集 第4卷",
            "series": "马克思恩格斯全集",
            "volume": "第4卷",
            "work_id": "german-ideology",
            "article": "德意志意识形态",
            "section": "费尔巴哈",
            "edition_id": "me-cn-2e",
            "publisher": "人民出版社",
            "publication_year": 1995,
            "retrieval_unit": "semantic_child",
            "parent_paragraph_id": "para-parent",
            "pdf_page_start": 12,
            "pdf_page_end": 13,
            "printed_page_start": None,
            "printed_page_end": "",
            "citation_page_start": 7,
            "citation_page_end": 8,
            "citation_page_type": "printed_page",
            "page_type": "body",
            "page_span": [12, 13],
            "source_page_ids": ["mea04.pdf#page-12", "mea04.pdf#page-13"],
            "text_source": "pdf_text",
            "content_class": "body",
            "quality_status": "passed",
            "build_id": "corpus-v2-test",
            "chunker_version": "semantic-child/v2",
            "chunker_config_hash": "cfg123",
            "quality_flags": ["cross_page"],
            "spans": [{"page_id": "pg_1", "char_start": 2, "char_end": 8}],
            "bibliography_confidence": {"work_id": 1.0},
            "cleaning_reasons": ["join_cross_page"],
            "paragraph_text": "完整权威段落文本",
            "child_chunk_index": 2,
            "child_chunk_total": 3,
            "child_char_start": 5,
            "child_char_end": 11,
        }

    def test_stable_id_is_deterministic_and_location_sensitive(self):
        record = self.sample_record()
        record["spans"] = [{"page_id": "pg_1", "char_start": 20, "char_end": 40}]
        first = canonical_record_id(record)
        self.assertEqual(first, canonical_record_id(dict(record)))
        self.assertTrue(first.startswith("mr2_"))
        changed = {**record, "child_char_start": 6}
        self.assertNotEqual(first, canonical_record_id(changed))
        renumbered = {**record, "paragraph_index": 999}
        self.assertEqual(first, canonical_record_id(renumbered))
        moved = {**record, "spans": [{"page_id": "pg_1", "char_start": 21, "char_end": 40}]}
        self.assertNotEqual(first, canonical_record_id(moved))
        self.assertEqual("explicit-id", canonical_record_id({**record, "record_id": "explicit-id"}))
        self.assertEqual("ret-explicit", canonical_record_id({**record, "retrieval_id": "ret-explicit"}))

    def test_row_maps_provenance_hashes_and_null_integer_boundary(self):
        record = self.sample_record()
        record["document_record_version"] = "document-record/v1"
        indexed = "权威段落"
        row = row_from_record_v2(record, [0.1, 0.2], indexed)
        self.assertEqual(CONTRACT_VERSION, row["document_record_version"])
        self.assertEqual("german-ideology", row["work_id"])
        self.assertEqual("me-cn-2e", row["edition_id"])
        self.assertEqual(-1, row["printed_page_start"])
        self.assertEqual(-1, row["printed_page_end"])
        self.assertEqual(text_sha256(record["paragraph_text"]), row["source_text_hash"])
        self.assertEqual(text_sha256(indexed), row["indexed_text_hash"])
        self.assertTrue(row["text_was_clipped"])
        self.assertEqual(0, row["indexed_char_start"])
        self.assertEqual(len(indexed), row["indexed_char_end"])
        self.assertEqual([12, 13], json.loads(row["page_span_json"]))
        self.assertEqual(record["source_page_ids"], json.loads(row["source_page_ids_json"]))
        self.assertEqual(record["cleaning_reasons"], json.loads(row["cleaning_reasons_json"]))
        self.assertEqual(record["spans"], json.loads(row["spans_json"]))
        self.assertEqual(record["quality_flags"], json.loads(row["quality_flags_json"]))
        self.assertEqual(record["bibliography_confidence"], json.loads(row["bibliography_confidence_json"]))
        self.assertEqual("body", row["content_class"])
        self.assertEqual("corpus-v2-test", row["build_id"])
        self.assertTrue(row["retrievable"])
        self.assertEqual(set(BASE_OUTPUT_FIELDS), set(row) - {"embedding"})

    def test_output_fields_are_schema_backed_and_exclude_dense_vector(self):
        client = _Client()
        create_v2_schema(client, "corpus_v2", 1024, data_type=_DataType)
        schema_names = {field[0] for field in client.schema.fields}
        self.assertTrue(set(BASE_OUTPUT_FIELDS).issubset(schema_names))
        self.assertNotIn("embedding", BASE_OUTPUT_FIELDS)

    def test_explicit_index_range_and_sparse_embedding_are_preserved(self):
        record = {
            **self.sample_record(),
            "indexed_char_start": 5,
            "indexed_char_end": 11,
        }
        sparse = {4: 0.5}
        row = row_from_record_v2(record, [0.1], "权威段落", sparse_embedding=sparse)
        self.assertEqual(5, row["indexed_char_start"])
        self.assertEqual(11, row["indexed_char_end"])
        self.assertEqual(sparse, row[OPTIONAL_SPARSE_FIELD])

    def test_integer_boundary_round_trip(self):
        self.assertEqual(-1, to_milvus_int(None))
        self.assertEqual(-1, to_milvus_int("bad"))
        self.assertEqual(42, to_milvus_int("42"))
        self.assertIsNone(from_milvus_int(-1))
        self.assertEqual(42, from_milvus_int(42))

    def test_schema_fields_match_output_contract_with_optional_sparse(self):
        client = _Client()
        created = create_v2_schema(client, "corpus_v2", 1024, data_type=_DataType, enable_sparse=True)
        schema_names = [field[0] for field in client.schema.fields]
        self.assertEqual(list(CORE_FIELD_NAMES) + ["embedding", OPTIONAL_SPARSE_FIELD], schema_names)
        self.assertEqual(list(BASE_OUTPUT_FIELDS) + [OPTIONAL_SPARSE_FIELD], list(created.output_fields))
        self.assertEqual(2, len(client.indexes.indexes))

    def test_existing_collection_is_not_recreated_without_drop(self):
        client = _Client(existing=True)
        result = create_v2_schema(client, "corpus_v2", 1024, data_type=_DataType)
        self.assertFalse(result.created)
        self.assertEqual([], client.dropped)


if __name__ == "__main__":
    unittest.main()
