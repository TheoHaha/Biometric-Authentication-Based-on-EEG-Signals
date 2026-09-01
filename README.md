# Βiometric identification based on signals from electroencephalography (EEG) sensors
## Abstract
Authentication in the sense of confirming personal identity is a popular topic in the
field of information technology. In this context, various methods have been
developed that utilize biometric data, such as face and fingerprints. This paper
explores an authentication method that utilizes electroencephalograms (EEG).

Specifically, after appropriate preprocessing of the EEG signals, a deep learning
network is trained on them in order to extract a function that can generate vector-like
fingerprints unique to each user from any EEG recording. This function is then used
to generate fingerprints for any user who wants to register or log in to the system.
When a user wants to log into the system, their EEG fingerprint is recorded and
compared with the fingerprint of the user they claim to be using a vector distance
function. If the distance is found to be less than a certain threshold, which varies
depending on the distance function, the authentication is successful and the user’s
identity is confirmed.

The deep learning networks tested were a convolutional network from a related
study by Bidgoly et al. and a residual network based on ResNet18. Before training, a
channel selection algorithm was applied that selected the best channels from
specific search spaces based on training performance. The search spaces examined
were the channels from the “10-20” system and a small subset of frontal channels.
The algorithm managed to reduce the channels to just 3, while in the ResNet’s case
experiments were carried out with even fewer channels, all without sacrificing
training performance.

The authentication system was tested with various distance metrics as well as with
“known” and “unknown” to the system users in order to simulate realistic operating
conditions. The experiments led to satisfactory results for the ResNet18 architecture,
with authentication accuracy reaching 97.8%. These results suggest ResNet as a
network architecture that can compete with others from similar research.

Read the full paper *[here](https://drive.google.com/file/d/16ebKucZU0kUu1V7j4KmWh01Wiq-nP21n/view?usp=sharing)*. (GR)

## Main Inspiration
Bidgoly, A.J., Bidgoly, H.J. & Arezoumand, Z. Towards a universal and privacy preserving EEG-based authentication system. Sci Rep 12, 2531 (2022). https://doi.org/10.1038/s41598-022-06527-7
