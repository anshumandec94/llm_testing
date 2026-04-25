This project serves as the testing playground for an agent-based modelling experiment.
Recently, LLMs have seen a boom in their usage in the research domain, including agent-based modelling.
Particularly, in the domain of recommender systems, LLMs have resulted in a re-emergence of interest in agent-based modelling systems.
Agent-based modelling systems gained popularity in RecSys domain mainly as a complex evaluation tool, uncovering recommendation issues
that manifest over longer duration of time. However, the deterministic simplicity of traditional simulators limited this interest. LLMs and the recent increase in computational power have changed that outlook. Yet, we do not know if LLMs are 'reliable' and 'verifiable' simulators. The only way to test this, is to explore the 'transformer' aspect that LLMs (and indeed transformer based models) introduce. This project aims to be that exploratory playground where we take a fixed use-case of agent-based modelling (recommender systems) and test what and where LLMs or transformers perform better than traditional methods.

Our recommendation task is in the domain of Movie recommendation. We will make use of MovieLens-32M, a very popular recommender-systems dataset.
We aim to compare the following types of agents
1. Traditional Agent (Associative Embeddings)
This agent evaluates movies using embeddings generated from historical interactions (here, ratings) of the user (their user vector) compared with the item vector of the candidate movies.
2. Traditional Agent (Semantic Embeddings)
This agent evaluates movies using the content embeddings of the historical movies interacted with and then compare the distance of the centroid of their tastes to the candidate item and return a utility_score.
3. Sequence to Sequence Agent (seq2seq prediction)
This agent is based on a sequence-to-sequence prediction model that we will have to train on ML-32M ratings dataset. 
(here is a tensorflow implementation https://github.com/kang205/SASRec ) or use zero-shot foundational model. 
The model is trained to predict sequences and
4. LLM Agent (pure LLM reasoning + memory retrieval)
This agent is based on using an LLM as the agent along with memory retriever to retrieve user memory to reason about future recommended candidates. These implementations can vary and thus be used to evaluate different methods of LLM implementation

The 3 main pieces here are the environment, agent and the recommender. 
The environment holds global embeddings including the associative embeddings for users (on held-in information i.e. not the test held-out set we will evaluate on), the semantic embeddings for each movie (based on the movie content info from movies.csv or TMDB),

This is the setup
    1. Environment -  The environment holds all the data that we are dealing with. It loads the different csvs that we are going to use to generate all kinds of embeddings. This includes loading ratings, movie_info and joining movies with external information like TMDB for movie overview and other information. (movielens provides movieId per movie in movies.csv for TMDB Movie ID so querying is straightforward to get)
    The job of the environment is to create and prepare the relevant embeddings that will be used by the agents and the recommender. 
    ChromaDB will be used as the vector store to hold embeddings and retrieve them as required.
    2. Agent -  The agent would hold an implementation based on the type of agent they are. But simply put, they can only perform the actions of selecting a list of movies from a list of movies and generating or providing ratings for each of the movies provided in a candidate list. (Should we allow the agent to rate, we would have to implement that choice behavior individually too). The agent will receive a set of recommendations, evaluate them (based on implementation) and act on them appropriately.
    3. Recommender - The recommender is representative of the dynamic algorithm that aims to provide users with the best recommendations. The recommender would hold an internal representation of each user (based on public rating history seen in the environment) which is updated at each step when the user returns feedback on the recommended list of items. The recommender would update it's internal representation of the user vector which is fixed in type independent of the agent. This is separate from the agent's internal representation and makes the simulation more like the real world where the recommender's perception of user preference is different from reality (given it is an approximation).

The Evaluation Framework:
You hold-out a subset of user-item ratings. We try to make sure we have a good mix of items that are well-represented otherwise in the training data (in terms of ratings) and then run a few rounds of simulations where we show users recommended items outside of their rating list (including what was held-out) and then check where the held-out items ended up and also what they were rated if at all. This would be the first level of evaluation and comparing performance of user models to get as many of the held-out ratings.


Environment:
1. Contains all ratings used for training set
1.1 contains associative embeddings using matrix factorization-like techniques. The base environment creates the recommender (based on public user ratings) as well as the embeddings to be used to model the users themselves i.e. to evaluate recommended items. The main thing the environment first needs to do is separate the user ratings to create a set of held-out ratings for a subset of users that are to be evaluated on. The environment will then track these marked users as well as create the new ratings data-frame that is missing the held-out ratings. The environment will also provide the embeddings based on only the "seen" ratings that will be used by the recommender.
2.1 Recommender - we will use the Lenskit python package to create and generate the necessary recommendation pipeline.