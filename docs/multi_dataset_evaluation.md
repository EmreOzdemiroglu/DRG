# Çoklu Dataset Değerlendirmesi: Domain Duyarlılık Analizi

## 1. Genel Bakış

Multi-dataset evaluation, pipeline'ın farklı domain'lerdeki performansını değerlendirir. 3-4 heterojen dataset üzerinde chunking quality, semantic retrieval accuracy ve entity extraction effectiveness analiz edilir.

## 2. Dataset Seçimi

### 2.1 Dataset Kriterleri

**Heterogeneity:**
- Farklı text length'leri
- Farklı structure'lar
- Farklı domain'ler
- Farklı writing styles

**Evaluation Coverage:**
- Her dataset farklı challenge'ları test etmeli
- Comprehensive evaluation için yeterli diversity

### 2.2 Seçilen Dataset'ler

#### 2.2.1 Long Narrative Text (20-page story)

**Örnek:**
- Science fiction short story
- ~20 pages, ~50,000 words
- Character-driven narrative
- Temporal progression

**Challenges:**
- **Temporal Continuity**: Zaman içinde ilerleyen olaylar
- **Character Consistency**: Karakterlerin tutarlılığı
- **Plot Progression**: Hikaye akışı
- **Context Window**: Uzun metin, context window limitleri

**Evaluation Focus:**
- Chunking quality: Character/plot boundary preservation
- Semantic retrieval: Temporal queries ("what happened after X?")
- Entity extraction: Character entity consistency

#### 2.2.2 Factual Text (Wikipedia biography)

**Örnek:**
- Wikipedia article: "Albert Einstein"
- ~5-10 pages, ~10,000 words
- Structured sections (Early life, Career, Achievements)
- Factual, encyclopedic style

**Challenges:**
- **Factual Accuracy**: Gerçek bilgilerin doğruluğu
- **Entity Consistency**: Entity'lerin tutarlılığı
- **Chronological Ordering**: Zaman sıralaması
- **Cross-References**: Diğer entity'lere referanslar

**Evaluation Focus:**
- Chunking quality: Section boundary preservation
- Semantic retrieval: Factual queries ("when did X happen?")
- Entity extraction: Entity relationship accuracy

#### 2.2.3 Technical/Structured Document

**Örnek:**
- API documentation
- Software architecture document
- Technical specification
- ~15-20 pages, ~8,000 words
- Code blocks, tables, diagrams

**Challenges:**
- **Code Block Preservation**: Code block'ların bütünlüğü
- **Table Structure**: Tablo yapısının korunması
- **Cross-References**: API endpoint'ler, function'lar
- **Technical Terminology**: Domain-specific jargon

**Evaluation Focus:**
- Chunking quality: Code block/table atomicity
- Semantic retrieval: Technical queries ("how to use API X?")
- Entity extraction: Code entity extraction (functions, classes)

#### 2.2.4 Informal Dialogue (Chat/Forum)

**Örnek:**
- Reddit discussion thread
- Slack conversation
- Forum post + replies
- ~100-200 messages, ~5,000 words
- Multi-party dialogue

**Challenges:**
- **Conversation Turn Preservation**: Mesaj sıralaması
- **Context Switching**: Konu değişimleri
- **Multi-Party Dialogue**: Birden fazla konuşmacı
- **Informal Language**: Günlük dil, kısaltmalar

**Evaluation Focus:**
- Chunking quality: Conversation turn boundary preservation
- Semantic retrieval: Context-aware queries ("what did user X say?")
- Entity extraction: User entity extraction, topic extraction

## 3. Evaluation Metrics

### 3.1 Chunking Quality Metrics

#### 3.1.1 Entity Boundary Preservation

**Metric:**
- Entity'lerin chunk boundary'lerinde kesilme oranı
- Target: < 5% entity boundary violation

**Measurement:**
1. Ground truth entity'leri extract et (manual annotation veya high-quality NER)
2. Chunk boundary'lerini kontrol et
3. Entity kesilme oranını hesapla

**Formula:**
```
violation_rate = (entities_cut_at_boundary / total_entities) * 100
```

**Domain-Specific Targets:**
- Narrative: < 3% (character names critical)
- Factual: < 5% (person names, locations)
- Technical: < 7% (function names, API endpoints)
- Dialogue: < 10% (user names, informal entities)

#### 3.1.2 Semantic Coherence

**Metric:**
- Chunk içi semantic similarity (intra-chunk)
- Chunk'lar arası semantic similarity (inter-chunk)
- Target: Intra-chunk > inter-chunk

**Measurement:**
1. Chunk'ları embed et
2. Intra-chunk similarity hesapla (chunk içi sentence'lar)
3. Inter-chunk similarity hesapla (komşu chunk'lar)
4. Ratio hesapla

**Formula:**
```
coherence_ratio = mean(intra_chunk_similarity) / mean(inter_chunk_similarity)
target: coherence_ratio > 1.2
```

#### 3.1.3 Token Distribution

**Metric:**
- Chunk size distribution
- Target: Mean ≈ target window size, Std < 20% of mean

**Measurement:**
- Token count histogram
- Statistical analysis (mean, std, min, max)

**Domain-Specific Targets:**
- Narrative: Mean 768, Std < 150
- Factual: Mean 768, Std < 120
- Technical: Mean 1024, Std < 200 (code blocks)
- Dialogue: Mean 512, Std < 100

### 3.2 Semantic Retrieval Accuracy

#### 3.2.1 Precision@K

**Metric:**
- Top-K chunk'lar içinde relevant olanların oranı
- Target: Precision@10 > 0.7

**Measurement:**
1. Query set'i oluştur (her dataset için 20-30 query)
2. Ground truth relevant chunk'ları annotate et
3. Retrieval yap, Precision@K hesapla

**Domain-Specific Targets:**
- Narrative: Precision@10 > 0.6 (temporal queries zor)
- Factual: Precision@10 > 0.8 (factual queries kolay)
- Technical: Precision@10 > 0.7 (technical queries orta)
- Dialogue: Precision@10 > 0.65 (context switching zor)

#### 3.2.2 Recall@K

**Metric:**
- Top-K chunk'lar içinde bulunan relevant chunk'ların tüm relevant chunk'lara oranı
- Target: Recall@10 > 0.6

**Domain-Specific Targets:**
- Narrative: Recall@10 > 0.5
- Factual: Recall@10 > 0.7
- Technical: Recall@10 > 0.6
- Dialogue: Recall@10 > 0.55

#### 3.2.3 MRR (Mean Reciprocal Rank)

**Metric:**
- İlk relevant chunk'ın rank'ının reciprocal'i
- Target: MRR > 0.8

**Domain-Specific Targets:**
- Narrative: MRR > 0.7
- Factual: MRR > 0.85
- Technical: MRR > 0.75
- Dialogue: MRR > 0.7

### 3.3 Entity Extraction Effectiveness

#### 3.3.1 Entity Extraction F1

**Metric:**
- Entity extraction precision, recall, F1
- Target: F1 > 0.8

**Measurement:**
1. Ground truth entity'leri annotate et
2. Extracted entity'leri al
3. Precision, recall, F1 hesapla

**Domain-Specific Targets:**
- Narrative: F1 > 0.75 (character names, locations)
- Factual: F1 > 0.85 (person names, dates, locations)
- Technical: F1 > 0.7 (function names, API endpoints)
- Dialogue: F1 > 0.65 (user names, informal entities)

#### 3.3.2 Relation Extraction F1

**Metric:**
- Relation extraction precision, recall, F1
- Target: F1 > 0.7

**Domain-Specific Targets:**
- Narrative: F1 > 0.6 (character relationships)
- Factual: F1 > 0.75 (biographical relationships)
- Technical: F1 > 0.7 (API dependencies)
- Dialogue: F1 > 0.5 (user interactions)

#### 3.3.3 Graph Completeness

**Metric:**
- Extracted graph'ın ground truth graph'e coverage'i
- Target: Coverage > 0.6

**Measurement:**
1. Ground truth graph oluştur (manual annotation)
2. Extracted graph'ı al
3. Node/edge coverage hesapla

## 4. Comparison Table

### 4.1 Chunking Quality Comparison

| Dataset | Entity Boundary Violation | Semantic Coherence | Token Distribution |
|---------|---------------------------|-------------------|-------------------|
| Narrative | < 3% | > 1.3 | Mean 768, Std < 150 |
| Factual | < 5% | > 1.2 | Mean 768, Std < 120 |
| Technical | < 7% | > 1.1 | Mean 1024, Std < 200 |
| Dialogue | < 10% | > 1.15 | Mean 512, Std < 100 |

### 4.2 Retrieval Accuracy Comparison

| Dataset | Precision@10 | Recall@10 | MRR |
|---------|--------------|-----------|-----|
| Narrative | > 0.6 | > 0.5 | > 0.7 |
| Factual | > 0.8 | > 0.7 | > 0.85 |
| Technical | > 0.7 | > 0.6 | > 0.75 |
| Dialogue | > 0.65 | > 0.55 | > 0.7 |

### 4.3 Entity Extraction Comparison

| Dataset | Entity F1 | Relation F1 | Graph Coverage |
|---------|-----------|------------|---------------|
| Narrative | > 0.75 | > 0.6 | > 0.5 |
| Factual | > 0.85 | > 0.75 | > 0.7 |
| Technical | > 0.7 | > 0.7 | > 0.6 |
| Dialogue | > 0.65 | > 0.5 | > 0.4 |

## 5. Domain Sensitivity Observations

### 5.1 Chunking Sensitivity

**Narrative Text:**
- Character name preservation critical
- Temporal continuity important
- Overlap strategy crucial (20% recommended)

**Factual Text:**
- Section boundary preservation important
- Entity consistency critical
- Balanced overlap (15% recommended)

**Technical Documents:**
- Code block atomicity critical
- Table structure preservation important
- Larger window size (1024 tokens)

**Dialogue:**
- Conversation turn preservation critical
- Context switching handling important
- Smaller window size (512 tokens)

### 5.2 Retrieval Sensitivity

**Narrative Text:**
- Temporal queries challenging
- Character-centric queries work well
- GraphRAG helps with character relationships

**Factual Text:**
- Factual queries work well
- Entity-centric queries excellent
- Classic RAG sufficient

**Technical Documents:**
- Technical terminology challenging
- Code-aware retrieval important
- Hybrid search (vector + keyword) helps

**Dialogue:**
- Context switching challenging
- Speaker attribution important
- Metadata filtering crucial

### 5.3 Entity Extraction Sensitivity

**Narrative Text:**
- Character names: High precision, medium recall
- Location names: Medium precision, medium recall
- Temporal entities: Low precision, low recall

**Factual Text:**
- Person names: Very high precision, high recall
- Dates: High precision, high recall
- Locations: High precision, high recall

**Technical Documents:**
- Function names: Medium precision, medium recall
- API endpoints: High precision, medium recall
- Technical terms: Low precision, low recall

**Dialogue:**
- User names: High precision, high recall
- Informal entities: Low precision, low recall
- Topic entities: Medium precision, medium recall

## 6. Failure Cases and Edge Behaviors

### 6.1 Chunking Failures

**Case 1: Entity Boundary Violation**
- **Scenario**: Character name chunk boundary'de kesilmiş
- **Impact**: Entity extraction fails, retrieval misses context
- **Mitigation**: Overlap strategy, boundary-aware chunking

**Case 2: Code Block Fragmentation**
- **Scenario**: Code block chunk boundary'de kesilmiş
- **Impact**: Code syntax broken, technical retrieval fails
- **Mitigation**: Atomic code block preservation

**Case 3: Table Fragmentation**
- **Scenario**: Table chunk boundary'de kesilmiş
- **Impact**: Table structure lost, information incomplete
- **Mitigation**: Atomic table preservation

### 6.2 Retrieval Failures

**Case 1: Semantic Drift**
- **Scenario**: Query semantic olarak farklı yorumlanmış
- **Impact**: Wrong chunks retrieved
- **Mitigation**: Query expansion, re-ranking

**Case 2: Entity Mismatch**
- **Scenario**: Query entity'si graph'da yok
- **Impact**: GraphRAG fails, fallback to RAG
- **Mitigation**: Entity normalization, fuzzy matching

**Case 3: Context Loss**
- **Scenario**: Multi-hop query, intermediate context lost
- **Impact**: Incomplete retrieval
- **Mitigation**: Path preservation, context window expansion

### 6.3 Entity Extraction Failures

**Case 1: Ambiguous Entities**
- **Scenario**: "Apple" (company vs fruit)
- **Impact**: Wrong entity type assignment
- **Mitigation**: Context-aware disambiguation

**Case 2: Rare Entities**
- **Scenario**: Domain-specific terminology
- **Impact**: Low recall
- **Mitigation**: Domain-specific schemas, fine-tuning

**Case 3: Informal Entities**
- **Scenario**: Chat/forum informal language
- **Impact**: Low precision, low recall
- **Mitigation**: Informal language handling, user entity extraction

## 7. Evaluation Protocol

### 7.1 Dataset Preparation

1. **Dataset Collection**: 3-4 heterojen dataset topla
2. **Annotation**: Ground truth entity'leri, relations'ları, relevant chunk'ları annotate et
3. **Query Generation**: Her dataset için 20-30 query oluştur
4. **Split**: Train/validation/test split (80/10/10)

### 7.2 Evaluation Execution

1. **Chunking**: Her dataset'i chunk'la, quality metrics hesapla
2. **Embedding**: Chunk'ları embed et, vector store'a ekle
3. **Graph Construction**: Entity/relation extraction, graph oluştur
4. **Retrieval**: Query'leri çalıştır, retrieval metrics hesapla
5. **Comparison**: RAG vs GraphRAG vs Hybrid karşılaştır

### 7.3 Reporting

1. **Metrics Table**: Tüm metrics'leri tablo halinde sun
2. **Domain Analysis**: Domain-specific observations
3. **Failure Analysis**: Failure cases ve mitigation strategies
4. **Recommendations**: Domain-specific recommendations

## 8. Recommendations by Domain

### 8.1 Narrative Text

**Chunking:**
- Window: 768 tokens
- Overlap: 20%
- Strategy: Sentence-aware, character entity tracking

**Retrieval:**
- Primary: GraphRAG (character relationships)
- Secondary: RAG (semantic queries)
- Hybrid: Reciprocal rank fusion

**Entity Extraction:**
- Schema: Character, Location, Event entities
- Focus: Character name consistency

### 8.2 Factual Text

**Chunking:**
- Window: 768 tokens
- Overlap: 15%
- Strategy: Section-aware, entity-centric

**Retrieval:**
- Primary: RAG (factual queries work well)
- Secondary: GraphRAG (entity relationships)
- Hybrid: Metadata-enhanced RAG

**Entity Extraction:**
- Schema: Person, Date, Location, Organization
- Focus: Factual accuracy

### 8.3 Technical Documents

**Chunking:**
- Window: 1024 tokens
- Overlap: 10%
- Strategy: Code-aware, table-aware

**Retrieval:**
- Primary: Hybrid (vector + keyword)
- Secondary: GraphRAG (API dependencies)
- Re-ranking: Cross-encoder

**Entity Extraction:**
- Schema: Function, Class, API, TechnicalTerm
- Focus: Code entity extraction

### 8.4 Dialogue

**Chunking:**
- Window: 512 tokens
- Overlap: 15%
- Strategy: Turn-aware, speaker-attribution

**Retrieval:**
- Primary: Metadata-enhanced RAG
- Secondary: GraphRAG (user relationships)
- Filtering: Speaker, topic, timestamp

**Entity Extraction:**
- Schema: User, Topic, Message
- Focus: User entity extraction, topic extraction

