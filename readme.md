# Recommender systems based on GPT

## Project Description

GPT-Recommender-systems is a project devoted to GPT-based news recommender systems (NRS). It tests ChatGPT as NRS and introduces three NRSs based on the OpenAI's text-embedding-ada-002: content-based, collaborative, hybrid. 

**Data** 

The data refers to the Microsoft News Dataset (MIND). The data for MIND was gathered from user behavior logs within Microsoft News. A random sample of one million users, each with a minimum of 5 news click records, was taken from October 12 to November 22, 2019. MIND contains around 160,000 news articles in English, accompanied by over 15 million impression records created by a user base of one million individuals. Each news article within MIND is equipped with comprehensive textual elements, including a title, abstract, body, category, and sub-category. In addition, it comprises click events (click and non-click) for each user, related to particular articles [1]. 

`behaviorsTEST.tsv` contains the table with three user profiles taken from MIND. 

`newsTEST.tsv` referes to the table with the information about news articles, users from `behaviorsTEST.tsv` interacted with.

**Calculating Embeddings** 

The `calculate_embeddings.ipynb` script coverts the title and abstract of news articles (`newsTEST.tsv`) into embeddings. 

**Creating embedding-based news recommender systems** 

`01_collaborative_recommender.ipynb` represents the mechanism for creating collaborative recommendations. The distances between embeddings related to each pair of user profiles are calculated using cosine similarity. For each user, the list of other users is generated and sorted, basing on user profile similarity, from shorter to longer distances. The final step relates to the distance normalization. The distances of articles for each user are normalized basing on similarity of input user with other users. The individial lists with sorted distances for each user are saved in `collaborative recommedations/`

`02_content_recommender.ipynb` introduces the process of creating content-based recommendations. The distances, expressed by a cosine similarity, between embeddings related to a user profile and embeddings referred to each of articles, not presented in the impression events of this user, are calculated. The individial lists with sorted distances for each user are saved in `content_recommedations/`

`03_hybrid_recommender.ipynb` shows how create hybrid news recommendations. For each user, two individual lists with article distances are generated: content-based and collaborative-based. Then, the mean distance is calculated for each news article in individual lists. The individial lists with sorted distances for each user are saved in `hybrid_recommedations/`

## Evaluation

**Performance metrics** 

`evaluation_general.ipynb` represents performance metrics such as precision, recall, F1 score.  The impression record referring to click events, with index “-1” indicating click, and index “-0” indicating no click, is considered.

**Diversity metrics** 

The data is annototed in terms of sentiment using the `TextBlob` package. The mechanism of it is demonstrated in `sentiment_analysis.ipynb`. The annotated news articles are saved in the folder `sentiment_analysis/`

`evaluation_diversity.ipynb` refers to the diversity metrics: content and sentiment. 

## Contributing
Contributions are welcome! If you'd like to contribute to this project, feel free to submit pull requests or open issues on the GitHub repository.

## Contact Information
If you have any questions, suggestions, or feedback, please feel free to reach out:

Email: kristina.barabashova1@gmail.com

GitHub: krisbara


[1] Fangzhao Wu, Ying Qiao, Jiun-Hung Chen, Chuhan Wu, Tao Qi, Jianxun Lian, Danyang Liu, Xing Xie, Jianfeng Gao, Winnie Wu, and others. 2020. Mind: A large- scale dataset for news recommendation. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics. 3597–3606.