# Recommendation-system-Connecting-business-users-with-innovative-solutions
Aim of this work was to explore possibility of automatically linking innovative
project proposals submitted via ENEL’s Open Innovability platform, with internal
organization, employees and teams, who would be able to provide relevant feedback
in the process of projects evaluation. Problem is formulated as a Natural Language
Processing task of finding similarity between texts, which are representing project
descriptions and employee profiles, resulting with a recommendation list of relevant
employees for a proposed project.
Based on characteristics of available datasets, which appeared in the process of
Exploratory Data Analysis, after appropriate text preparation, we proposed several
methods.<br>Firstly, we applied Latent Dirlecht Allocation (LDA) method to represent
project descriptions in form of topic probability and quantified similarity between
projects and employees in terms of weights assigned to common words, appearing
among representative topics’ words and descriptions of employees. <br>Secondly, we
experimented with word and document representations in vector space. For creating
word vector embeddings we used well-known neural network models Word2Vec,
GloVe and fastText. Similarity project - employee is inversely proportional to
Word’s Mover Distance (WMD) between them. Document vector representations
were created in two ways, as Smooth Inverse Frequency (SIF) of word vectors and
applying Paragraph vectors (Doc2Vec) model and text similarity was expressed in
terms of cosine similarity between associated vectors. <br>Thirdly, specialized method
for finding similarity between documents of significantly different lengths [16] was
used. Method is based on finding hidden topic vectors and their relevance within
project descriptions and calculating produced error when using those vectors for
reconstruction of employee description texts.<br>
In absence of previous knowledge about correct linking between project and
employee entities, for measuring results we proposed simple user-centric framework
which implies gathering feedback from Open Innovation Team, group of ENEL
employees in charge of managing innovation projects. They scored relevance of
suggestions generated by proposed models on a small sample. The first model LDA
with custom created similarity showed the most potential, followed by the third
model of text matching based on hidden topic vectors. Text vector representations
didn’t perform well.
