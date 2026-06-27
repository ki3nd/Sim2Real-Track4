2026 Challenge Track Description
Track 4: Text-Based Person Re-Identification (Sim2Real)

Text-based person re-identification aims to retrieve specific individuals across camera networks using natural language descriptions. However, current benchmarks often exhibit biases towards common actions like walking or standing, neglecting the critical need for identifying abnormal behaviors in real-world scenarios. To meet such demands, we propose a new task, text-based person anomaly search, locating pedestrians engaged in either routine or anomalous activities via text. To enable the training and evaluation of this new task, we construct a large-scale image-text Pedestrian Anomaly Behavior (PAB) benchmark, featuring a broad spectrum of actions (1,000 types), e.g., running, performing, playing soccer, and the corresponding anomalies (1,600 types), e.g., lying, being hit, and falling of the same identity. 

Data 
The PAB dataset is published in ICCV25. We adopt the same training set as used in ICCV25 and a new test set with distractors. 

The PAB training set consists of 1,013,605 synthetic images of normal and abnormal behaviors. Each image is accompanied by detailed textual descriptions of the target pedestrian’s appearance, action, and surrounding scene, as well as annotations of the normal/abnormal category and scene. 

The new test set (name-masked test set) includes 1,978 query texts (normal text: abnormal text = 1:1) and a gallery of real images. The gallery contains 1,978 ground-truth pedestrian images corresponding to the queries, along with 34,795 distractor images.

Please note that the PAB test set (query/gallery) cannot be used for training in any way. 

Evaluation Metric
For the leaderboard, the Mean Average Precision (mAP) will be employed. mAP refers to the average area under the precision-recall curve across all queries.

Important note: the test data and its distribution must not be used in any form during the training process. Using the official test set as a validation set—even without using ground-truth labels—is not permitted. The test set should be used strictly for final inference and leaderboard evaluation only. Any use of test set outputs (with or without labels) for model selection, threshold tuning, ensemble selection, pseudo-labeling, or post-processing adjustment is prohibited.

Submission Format
Teams should submit an “answer.txt” file containing the top-10 person image names for each query. The submission example can be found at “answer.txt”. 

For example, the first query index in “query_index.txt” is “IGMVCNHLXVJKBLN”. Therefore, the first line of the returned result in “answer.txt” should be formatted as follows from Rank-1 to Rank-10:

ZOVZW5GHWX3K7R2 OT6C4QNNMQFAF0J LLWPRL2M86UM0TN 4QV8E6SPPOSHMR3 PRL2M86UM0TNW4S BZ9QQ9FO7L7N0I0 7M374DFP45BIFZA N1EA3PVVUYNIN8H 03L3B7JHXCTOLT5 P20G5CGDLXYDEK1
The “answer.txt” we provide only displays the format and is not a real instance of search results

Additional Datasets 
Teams aiming for the public leaderboard and challenge awards must not use non-public datasets in any training, validation, or test sets. Winners and runners-up must submit their training and testing codes for verification after the deadline to confirm no non-public data was used, ensuring tasks were algorithm-driven, not human-performed.

Data Access
AIC26 Track4 data can be accessed here.