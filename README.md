See my blog!!!
https://becominghuman.ai/probabilistic-approaches-to-combinatorial-optimization-2aa0397a795f

Combinatorial optimization algorithms are designed to find an optimal object from a finite set of objects. In this article, I will examine two probabilistic techniques to solve such a problem: Genetic Algorithms (GA) and Simulated Annealing (SA). To demonstrate these two approaches, I will examine the raison d?être of combinatorial problems, namely, the Travelling Salesman Problem (TSP). The TSP asks the following question: ?Suppose you are given a list of cities and the distances between all these cities, what is the shortest path that visits each city exactly once and returns to the starting city??. The TSP is considered to be NP-hard, which means that it can?t be solved in polynomial time. That said, we can get a ?close enough? solution using randomized improvement with clever heuristics.

To launch the notebook, run the following command from a terminal with anaconda3 installed and on the application path:

    jupyter notebook AIND-Simulated_Annealing.ipynb
