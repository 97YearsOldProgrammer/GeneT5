# Training Species Survey

Cross-program comparison of species used for gene finder training and benchmarking

## GeneT5 Current Species (31)

| Taxa | Species |
|------|---------|
| Prokaryotes (6) | E.coli, B.subtilis, C.crescentus, Synechocystis.PCC6803, H.salinarum, V.fischeri |
| Unicellular (6) | S.cerevisiae, S.pombe, C.reinhardtii, N.crassa, D.discoideum, T.thermophila |
| Invertebrates (7) | C.elegans, D.melanogaster, N.vectensis, S.purpuratus, H.vulgaris, A.mellifera, B.mori |
| Vertebrates (9) | C.jacchus, G.gallus, C.porcellus, X.tropicalis, H.sapiens, O.latipes, M.musculus, R.norvegicus, D.rerio |
| Plants (5) | P.patens, Z.mays, M.truncatula, O.sativa, A.thaliana |

## Programs Surveyed

| Program | Type | Species Count | Source |
|---------|------|---------------|--------|
| Augustus | HMM | 73+ pretrained | github.com/Gaius-Augustus/Augustus |
| BRAKER3 | Augustus+GeneMark pipeline | 11 benchmark | Genome Research 2024 |
| GALBA | Augustus+miniprot pipeline | 14 benchmark | BMC Bioinformatics 2023 |
| GeneMark-ETP | HMM (self-training) | 7 benchmark | Genome Research 2024 |
| Helixer | BiLSTM+HMM | ~290 genomes (4 clade models) | Nature Methods 2025 |
| Tiberius | CNN+LSTM+HMM | 37 mammalian + cross-species | Bioinformatics 2024 |
| GlimmerHMM | GHMM | 6+ trained | CCB JHU |
| SNAP | HMM | 4 shipped | Korf 2004 |
| Prodigal | prokaryote (self-training) | 10 benchmark | BMC Bioinformatics 2010 |
| Funannotate | fungal pipeline (Augustus/SNAP/GeneMark) | Augustus fungal models | funannotate.readthedocs.io |

## Cross-Program Consensus: Missing Species

### Tier 1 -- Used by 4+ programs (highest priority)

| Species | Common Name | Programs | Gap Filled |
|---------|-------------|----------|------------|
| Solanum lycopersicum | tomato | BRAKER3, GALBA, GeneMark-ETP, Augustus | Solanaceae (no nightshades) |
| Populus trichocarpa | black cottonwood | BRAKER3, GALBA, Augustus | woody plants (no trees) |

### Tier 2 -- Used by 2-3 programs

| Species | Common Name | Programs | Gap Filled |
|---------|-------------|----------|------------|
| Bombus terrestris | bumblebee | BRAKER3, GALBA | Hymenoptera diversity |
| Parasteatoda tepidariorum | house spider | BRAKER3, GALBA | entire Chelicerata |
| Tribolium castaneum | red flour beetle | Augustus, benchmarks | entire Coleoptera |
| Brugia malayi | filarial nematode | Augustus, GlimmerHMM | parasitic nematodes |
| Staphylococcus aureus | MRSA | Augustus, prokaryote benchmark | Gram+ pathogens |
| Pseudomonas aeruginosa | -- | Prodigal, prokaryote studies | high-GC Gram- |
| Cryptococcus neoformans | -- | Augustus, GlimmerHMM | Basidiomycete pathogens |
| Bos taurus | cow | Tiberius test species | large mammals |

### Tier 3 -- Single program but fills critical phylogenetic gap

| Species | Common Name | Program | Gap Filled |
|---------|-------------|---------|------------|
| Ciona intestinalis | sea squirt | Augustus | basal chordates (tunicates) |
| Branchiostoma floridae | lancelet | Augustus | cephalochordates |
| Tetraodon nigroviridis | pufferfish | GALBA | compact fish genome |
| Rhodnius prolixus | kissing bug | GALBA | entire Hemiptera |
| Nasonia vitripennis | jewel wasp | Augustus | parasitoid wasps |
| Toxoplasma gondii | -- | Augustus | entire Apicomplexa |
| Plasmodium falciparum | malaria | Augustus | AT-rich parasite |
| Schistosoma mansoni | blood fluke | Augustus | entire Platyhelminthes |
| Leishmania tarentolae | -- | Augustus | entire Kinetoplastida |
| Trichoplax adhaerens | placozoan | Augustus | basal metazoan |
| Amphimedon queenslandica | sponge | Augustus | Porifera |
| Vitis vinifera | grapevine | Augustus | eudicot crop diversity |
| Triticum aestivum | wheat | Augustus | hexaploid challenge |
| Aspergillus nidulans | model mold | Augustus, Funannotate | filamentous fungi |
| Mycobacterium tuberculosis | TB | Prodigal | Actinobacteria (high-GC) |
| Sulfolobus solfataricus | -- | Prodigal | Crenarchaeota |
| Mycoplasma genitalium | -- | prokaryote benchmark | minimal genome edge case |
| Magnaporthe grisea | rice blast | Augustus | plant pathogenic fungi |
| Aedes aegypti | mosquito | Augustus | Diptera diversity / vector |
| Petromyzon marinus | sea lamprey | Augustus | jawless vertebrates |

## Phylogenetic Gap Analysis

Entire taxonomic groups with zero GeneT5 representation:

| Group | Example Species to Add |
|-------|----------------------|
| Crenarchaeota | Sulfolobus solfataricus |
| Actinobacteria | Mycobacterium tuberculosis |
| Apicomplexa | Toxoplasma gondii, Plasmodium falciparum |
| Kinetoplastida | Leishmania tarentolae |
| Platyhelminthes | Schistosoma mansoni |
| Chelicerata | Parasteatoda tepidariorum |
| Coleoptera | Tribolium castaneum |
| Hemiptera | Rhodnius prolixus |
| Tunicata | Ciona intestinalis |
| Cephalochordata | Branchiostoma floridae |
| Agnatha (jawless fish) | Petromyzon marinus |
| Pathogenic fungi | Aspergillus nidulans, Cryptococcus neoformans |
| Solanaceae | Solanum lycopersicum |
| Trees | Populus trichocarpa |

## Top 20 Recommended Additions (Priority Order)

| # | Species | Rationale |
|---|---------|-----------|
| 1 | Solanum lycopersicum | 4-program consensus; Solanaceae gap; major benchmark |
| 2 | Populus trichocarpa | 3-program consensus; first tree genome; woody plant gap |
| 3 | Tribolium castaneum | Augustus + benchmarks; entire Coleoptera gap |
| 4 | Parasteatoda tepidariorum | BRAKER3 + GALBA; entire Chelicerata gap |
| 5 | Bombus terrestris | BRAKER3 + GALBA; Hymenoptera diversity |
| 6 | Bos taurus | Tiberius test; large mammal gap |
| 7 | Ciona intestinalis | Augustus; basal chordate key position |
| 8 | Aspergillus nidulans | Augustus + Funannotate; filamentous fungi gap |
| 9 | Staphylococcus aureus | Augustus + benchmark; Gram+ pathogen gap |
| 10 | Pseudomonas aeruginosa | Prodigal + benchmark; high-GC Gram- gap |
| 11 | Toxoplasma gondii | Augustus; entire Apicomplexa gap |
| 12 | Schistosoma mansoni | Augustus; entire Platyhelminthes gap |
| 13 | Cryptococcus neoformans | Augustus + GlimmerHMM; basidiomycete pathogen |
| 14 | Mycobacterium tuberculosis | Prodigal; Actinobacteria gap; high-GC |
| 15 | Sulfolobus solfataricus | Prodigal; Crenarchaeota gap |
| 16 | Vitis vinifera | Augustus; eudicot crop diversity |
| 17 | Triticum aestivum | Augustus; hexaploid genome challenge |
| 18 | Aedes aegypti | Augustus; Diptera diversity; disease vector |
| 19 | Branchiostoma floridae | Augustus; basal chordate evolution |
| 20 | Magnaporthe grisea | Augustus; plant pathogenic fungus |

## Scale Comparison

| Program | Training Species |
|---------|-----------------|
| GeneT5 (current) | 31 |
| Augustus | 73+ |
| Helixer | ~290 genomes |
| Tiberius | 37 mammals + cross-species tests |
| BRAKER3 benchmark | 11 |
| GALBA benchmark | 14 |
