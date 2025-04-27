https://www.youtube.com/watch?v=IShRYPsmiR8

# Nodes
CREATE (n) 

MATCH (n)

CREATE (n:Person {name: 'John Doe', age: 30, city: 'New York'})

MATCH (n:Person) RETURN n

Can you use Cypher using networkx? 

# Relationships

 MATCH (a:Person), (b:Person )
 WHERE a.name = 'John Doe' AND b.name = 'Jane Smith'
 CREATE (a)-[:FRIENDS_WITH]->(b)


# Examples (representations of data)

Software supply chain example

LLM to build cycle graph representation.

For eaach node can add a bunch of properties.
i.e contex for llm

properties on the relationships?

# query examples

get only nodes called a certain name
