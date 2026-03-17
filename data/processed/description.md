Androids Corpus – Processed Files

androids_folds_tidy.csv
Reshaped cross-validation fold assignments in long format, linking each recording to a fold and speech type (read or interview) for reproducible model evaluation.

androids_interview_timedata_clean.csv
Cleaned interview timing data containing timestamp boundaries for each recording, enabling potential segmentation or fine-grained temporal analysis of speech.

androids_metadata.csv
Comprehensive file-level metadata combining audio paths, BDI scores, depression labels, fold assignments, speech type, and interview timing information.

androids_metadata_core.csv
Simplified metadata table containing only essential fields (file path, labels, fold, speech type), used as the primary input for feature extraction and modelling.

RADAR-MDD – Processed Files

participant_labels_latest.csv
Participant-level dataset where each individual is assigned their most recent PHQ-8 score, providing a single-label summary for baseline analyses.

participant_labels_mean.csv
Participant-level dataset with PHQ-8 scores averaged across all questionnaire completions, representing overall depression severity per individual.

participant_labels_max.csv
Participant-level dataset where each individual is assigned their maximum observed PHQ-8 score, capturing peak depression severity.

questionnaire_phq8_clean.csv
Cleaned longitudinal dataset containing all PHQ-8 questionnaire instances with timestamps, computed total scores, and severity labels; used as the primary source of depression labels for time-aware analyses.