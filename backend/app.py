from backend.ai_interview_analyser import *
from sys import getsizeof
from flask_session import Session
import os
from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor
import os
from datetime import datetime
from functools import wraps
from types import SimpleNamespace
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity



load_dotenv()

app = Flask(__name__)
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_FILE_DIR"] = os.path.join(os.getcwd(), "flask_session")

Session(app)
model = SentenceTransformer('all-MiniLM-L6-v2')
model.save("models/interview_model")

nlp_model = SentenceTransformer("models/interview_model")

app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-me')

DATABASE_URL = os.environ.get('DATABASE_URL')
def get_db():
    if not DATABASE_URL:
        raise RuntimeError('DATABASE_URL not set in .env')
    return psycopg2.connect(DATABASE_URL)


def init_db():
    with get_db() as conn:
        with conn.cursor() as cur:

           
            cur.execute('''
                CREATE TABLE IF NOT EXISTS accounts (
                    id          SERIAL PRIMARY KEY,
                    first_name  VARCHAR(100),
                    last_name   VARCHAR(100),
                    email       VARCHAR(100) UNIQUE,
                    password    TEXT,
                    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

         
            cur.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id         SERIAL PRIMARY KEY,
                    name       VARCHAR(100),
                    email      VARCHAR(100) UNIQUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

           
            cur.execute('''
                CREATE TABLE IF NOT EXISTS questions (
                    id  SERIAL PRIMARY KEY,
                    category      VARCHAR(100),
                    question_text VARCHAR(2000),
                    ideal_answer  VARCHAR(5000),
                    key           VARCHAR(500),
                    difficulty    VARCHAR(20) DEFAULT 'medium'
                )
            ''')

            # interview sessions table
            cur.execute('''
                CREATE TABLE IF NOT EXISTS interview_session (
                    id             SERIAL PRIMARY KEY,
                    user_id        INTEGER REFERENCES users(id),
                    interview_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    overall_score  FLOAT,
                    category       VARCHAR(100),
                    total_questions INTEGER DEFAULT 10
                )
            ''')

            # your performance table (DataGrip)
            cur.execute('''
                CREATE TABLE IF NOT EXISTS performance (
                    performance_id SERIAL PRIMARY KEY,
                    id             INTEGER REFERENCES users(id),
                    category       VARCHAR(100),
                    average_score  FLOAT
                )
            ''')

            # user answers per question
            cur.execute('''
                CREATE TABLE IF NOT EXISTS user_answers (
                    answer_id    SERIAL PRIMARY KEY,
                    session_id   INTEGER,
                    user_id      INTEGER REFERENCES users(id),
                    question_id  INTEGER REFERENCES questions(id),
                    user_answer  TEXT,
                    score        FLOAT DEFAULT 0,
                    answered_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

        conn.commit()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT setval(pg_get_serial_sequence('interview_session','id'), COALESCE(MAX(id), 0) + 1, false) FROM interview_session"
            )
        conn.commit()
    seed_questions()
"""
SEED_DATA — 50 Interview Questions with Recruiter-Style Ideal Answers
Optimized for: paraphrase-MiniLM-L6-v2 cosine similarity scoring
Format: Rich, keyword-dense, structured answers matching real interviewer expectations.
"""

SEED_DATA = {

    # ══════════════════════════════════════════════════════
    # HR — BEHAVIOURAL & SOFT SKILLS
    # ══════════════════════════════════════════════════════
    'hr': [

        (
            "Tell me about yourself.",
            "I am a pre-final year B.Tech student specializing in Artificial Intelligence and Data Science with hands-on experience in machine learning NLP and full-stack web development. Currently I am building end-to-end AI applications using Flask PostgreSQL and Python and have completed a machine learning internship where I worked on real datasets and deployed production models. In the past I have participated in national hackathons building projects like a phishing detection system a cybersecurity threat analyzer and an AI financial assistant winning recognition for technical depth and innovation. Going forward I am looking to join a team where I can contribute my AI and data skills to solve meaningful real-world problems while growing as an engineer. I am passionate about building things that work not just things that look good on a slide.",
            "Use Present-Past-Future structure. State your role, achievements, and why you want this position."
        ),

        (
            "Where do you see yourself in 5 years?",
            "In five years I see myself working as a senior AI or machine learning engineer leading end-to-end development of intelligent systems that solve high-impact business problems. I want to move from building individual models to architecting entire ML pipelines mentoring junior engineers and contributing to product decisions at a strategic level. I plan to deepen expertise in large language models MLOps and scalable AI deployment. I am particularly excited about growing within a company that gives engineers real ownership over the products they build. My goal is not just a title change but a genuine expansion of technical depth and real-world impact.",
            "Show ambition, technical direction, and alignment with the company's growth goals."
        ),

        (
            "What is your greatest strength?",
            "My greatest strength is the ability to quickly learn and apply new technologies to solve practical problems under real constraints and tight deadlines. For example during a 48-hour hackathon I picked up the Google Gemini API for the first time designed a cybersecurity threat analysis agent built a Flask backend and deployed it to Render within the deadline. The project won recognition for both technical quality and real-world applicability. This combination of fast learning practical execution and staying calm under pressure has been consistently my strongest asset across internships projects and competitions.",
            "Name one genuine strength and support it with a specific measurable example."
        ),

        (
            "What is your greatest weakness?",
            "My biggest weakness is that I sometimes spend too much time perfecting a solution when a working version would have been sufficient to move forward. I tend to over-engineer edge cases before validating the core idea. I have been actively working on this by adopting a build-measure-learn approach shipping a minimum viable version first and iterating based on real feedback. For example during my AI Study Planner project I forced myself to deploy a basic version in the first week and only added advanced NLP features after confirming the core workflow worked correctly. This habit has made me significantly faster and more outcome-focused.",
            "Be honest, name a real weakness, and clearly demonstrate what you are doing to improve."
        ),

        (
            "Why do you want to work here?",
            "I want to work here because your company sits at the intersection of applied AI and real-world product impact which is exactly where I want to grow my career. I have followed your work in machine learning and I am impressed by how your engineering culture balances research with practical deployment at scale. My experience building NLP systems Flask APIs and deploying ML models gives me a strong foundation to contribute from day one while your team's scale and problem complexity would push me to grow faster than anywhere else. I am not looking for a comfortable role I am looking for one where the problems are genuinely hard and the team is strong enough to tackle them.",
            "Be specific about the company. Connect your actual skills to their real work."
        ),

        (
            "Describe a challenge you faced and how you overcame it.",
            "During my machine learning internship I was tasked with building a fraud detection model on a highly imbalanced dataset where fraudulent transactions made up less than one percent of all records. Initial models achieved high accuracy but completely failed to detect actual fraud because they simply predicted the majority class every time. I identified the root cause as class imbalance affecting the wrong optimization target. I applied SMOTE oversampling to balance the training data switched the evaluation metric to F1-score and AUC-ROC and tuned a Random Forest classifier using stratified cross-validation. The final model improved fraud recall by over 40 percent while maintaining acceptable precision making it genuinely useful for the business. The core lesson was that the right metric matters more than a good-looking accuracy number.",
            "Use STAR: Situation Task Action Result. Quantify the outcome wherever possible."
        ),

        (
            "Tell me about a time you worked in a team.",
            "In the CodHer 2026 hackathon I worked in a three-person team to build FairScan AI a system that detects algorithmic bias in machine learning models. I took ownership of the backend ML pipeline implementing fairness metrics like demographic parity and equal opportunity using the Fairlearn library while my teammates handled the frontend interface and presentation. We faced a critical conflict midway when the frontend and backend data formats did not match requiring us to sit together at midnight and redesign the API contract. By communicating clearly and prioritizing the working demo over individual preferences we delivered a complete integrated product on time and received strong feedback from judges on both technical depth and practical relevance.",
            "Show your specific role, a conflict or obstacle encountered, and how the team resolved it together."
        ),

        (
            "How do you handle pressure and tight deadlines?",
            "When facing pressure and tight deadlines I immediately break the overall goal into the smallest set of tasks that would produce a working result and then focus exclusively on those before adding anything else. During the Google Gemini Live Agent Challenge I had less than 30 hours to build ThreatLens AI a live cybersecurity analysis agent. I listed the three core features needed for a functional demo completed those first and only added polish in the final hours once the core was stable and tested. I also use time-boxing giving each task a fixed window and moving on when it ends which prevents me from getting stuck. Pressure actually sharpens my focus because it forces clear decisions about what truly matters.",
            "Give a concrete real example. Show a specific system or method you actually use under pressure."
        ),

        (
            "How do you prioritise your work?",
            "I prioritise work using a combination of impact and urgency asking two questions for every task: what is the cost of not doing this today and what value does it create if I do it now. High-impact and time-sensitive tasks go first. Low-impact tasks get batched or deferred. I maintain a weekly goal list and a daily task list reviewing them every morning so I always know my top three must-complete items. During my 100 Days AI Challenge I used this system to consistently ship one working project per day alongside college coursework by always knowing exactly what the minimum viable output for that day was and refusing to let secondary tasks expand into time reserved for primary ones.",
            "Mention a real system you use and demonstrate that you are structured and proactive not reactive."
        ),

        (
            "Do you have any questions for us?",
            "Yes I have a few thoughtful questions. First what does the first 90 days look like for someone joining this team and what would a successful contribution look like by the end of that period? Second how does your engineering team balance shipping product features quickly with maintaining code quality and managing technical debt? Third what is the biggest unsolved technical challenge your AI or data team is currently working through and where do you see the most opportunity for someone new to make an impact? I ask these because I want to understand how I can contribute meaningfully from day one and where the most interesting problems are.",
            "Always prepare 2 to 3 genuine informed questions. Show you have done your research."
        ),

    ],

    # ══════════════════════════════════════════════════════
    # PYTHON — CORE LANGUAGE & PROGRAMMING
    # ══════════════════════════════════════════════════════
    'python': [

        (
            "What is a decorator in Python?",
            "A decorator in Python is a higher-order function that takes another function as input wraps it with additional behaviour and returns the modified function without changing the original function's source code. Decorators use the at-sign syntax applied directly above a function definition. Writing at-decorator above a function is syntactic sugar equivalent to writing function equals decorator of function after the definition. Common real-world uses include enforcing login requirements on web routes logging function calls for debugging caching expensive results with functools.lru_cache measuring execution time and Flask route registration. Decorators follow the open-closed principle extending functionality without modifying the original code making them a powerful tool for clean separation of concerns.",
            "Explain what it does how it works syntactically and give at least two real use cases."
        ),

        (
            "What is the difference between a list and a tuple?",
            "The fundamental difference between a list and a tuple in Python is mutability. Lists are mutable meaning their elements can be added removed or changed after creation using methods like append remove and pop. Tuples are immutable meaning once created their contents cannot be modified. Lists use square brackets and tuples use parentheses. Because tuples are immutable they are hashable and can be used as dictionary keys or set elements whereas mutable lists cannot. Tuples are also slightly faster than lists for iteration and access because Python can make memory optimizations knowing the size will not change. Use lists when you need a collection that will grow or change and use tuples when representing a fixed group of values that logically belong together such as coordinate pairs database rows or function return values with multiple components.",
            "Cover mutability hashability use cases and performance differences clearly."
        ),

        (
            "What is a lambda function?",
            "A lambda function is an anonymous single-expression function defined using the lambda keyword in Python. Unlike regular functions defined with def a lambda has no name and consists of a single expression whose evaluated value is automatically returned. The syntax is lambda arguments colon expression. Lambda functions are most useful as short throwaway functions passed as arguments to higher-order functions. For example sorted of students with key equals lambda s colon s.grade sorts a list of student objects by their grade without defining a separate named function. Similarly map of lambda x colon x squared comma numbers applies squaring to every element. For anything more complex than a single expression a regular named function is clearer more readable and more maintainable.",
            "Define it show the syntax give two real examples and explain when to prefer a named function instead."
        ),

        (
            "Explain Python's memory management.",
            "Python manages memory automatically through a combination of reference counting and a cyclic garbage collector. Reference counting tracks how many variables or containers point to each object. When an object's reference count drops to zero its memory is immediately reclaimed without waiting. However reference counting alone cannot handle circular references where two objects reference each other but nothing external points to either. Python's cyclic garbage collector which is the gc module periodically scans for such cycles and frees them. All Python objects are stored in a private heap managed by the Python Memory Manager which uses a system of memory pools and arenas to efficiently handle frequent small allocations common in Python programs. Developers can use tracemalloc and objgraph to profile memory usage and detect leaks in long-running applications.",
            "Cover reference counting circular reference limitation garbage collector the heap and profiling tools."
        ),

        (
            "What is the GIL?",
            "The Global Interpreter Lock or GIL is a mutex in CPython that ensures only one thread executes Python bytecode at any given time even on multi-core processors. It exists because CPython's memory management uses reference counting which is not thread-safe without synchronization. The GIL prevents two threads from simultaneously modifying reference counts and corrupting memory. The practical consequence is that CPU-bound multi-threaded Python programs do not achieve true parallelism because only one thread can run Python code at a time. However the GIL is released during I/O operations so I/O-bound programs like web servers and network scrapers can still benefit from threading. For true CPU parallelism use the multiprocessing module which spawns separate processes each with their own Python interpreter and GIL. Alternatively Jython and PyPy implementations handle thread safety differently and do not have the same GIL limitation.",
            "Explain what it is why it exists its impact on CPU vs IO bound tasks and how to work around it."
        ),

        (
            "What is the difference between deepcopy and copy?",
            "Python's copy module provides two functions that create copies of objects but differ fundamentally in how they handle nested objects. copy.copy creates a shallow copy which creates a new top-level container object but the elements inside still reference the same objects as the original. Modifying a mutable nested object such as a list inside the copy also modifies it in the original because both the original and the copy point to the same inner object. copy.deepcopy creates a deep copy which recursively duplicates every object in the entire hierarchy creating completely independent copies at every level. Changes to any part of the deep copy have no effect whatsoever on the original. Use shallow copy when sharing inner objects is safe or intentional. Use deep copy when you need a fully independent clone such as when implementing undo functionality cloning game states or processing objects in parallel without interference.",
            "Explain both types clearly demonstrate the difference with nested objects and give practical use cases."
        ),

        (
            "What are *args and **kwargs?",
            "In Python function definitions *args and **kwargs allow functions to accept a variable number of arguments making function signatures flexible. *args collects any extra positional arguments beyond those explicitly named into a tuple. The function can then iterate over this tuple to process each additional argument. **kwargs collects any extra keyword arguments into a dictionary where each key is the argument name and each value is the provided value. Both are commonly used when writing decorator functions that wrap other functions and need to pass all arguments through unchanged. For example a logging decorator uses *args and **kwargs to call the wrapped function with whatever arguments the caller provided while adding log statements before and after. They are also essential in class inheritance for forwarding arguments to parent class constructors. The names args and kwargs are conventional but not required. Only the asterisks carry syntactic meaning.",
            "Explain what each collects the data type returned and give a real practical use case for each."
        ),

        (
            "What is list comprehension?",
            "List comprehension is a concise and readable Python syntax for creating a new list by applying an expression to each element of an iterable with an optional filter condition all in a single line. The syntax is opening bracket expression for item in iterable if condition closing bracket. For example x squared for x in range of ten if x modulo two equals zero creates a list of squared values for all even numbers from zero to nine. List comprehensions are generally faster than equivalent for loops with append because they are optimized at the bytecode level and avoid repeated attribute lookup on the append method. Python also supports dictionary comprehensions set comprehensions and generator expressions using similar syntax. Generator expressions use parentheses instead of brackets and produce values lazily without creating the entire list in memory making them preferable for large sequences. However deeply nested or overly complex comprehensions reduce readability at which point a regular for loop is the better choice.",
            "Show the syntax a concrete example the performance benefit and when to prefer a loop instead."
        ),

        (
            "Explain the four pillars of OOP in Python.",
            "Object-oriented programming in Python rests on four core principles. Encapsulation means bundling data and the methods that operate on it within a class and restricting direct access to internal state using private or name-mangled attributes. This protects data integrity and hides implementation details from external code. Abstraction means exposing only the necessary interface to the outside world while hiding the underlying complexity. Abstract base classes in Python formalize this through the abc module requiring subclasses to implement specific methods. Inheritance means a child class derives attributes and methods from a parent class enabling code reuse and hierarchical organization. Python supports multiple inheritance and uses the method resolution order to determine which parent's method takes precedence. Polymorphism means different classes can implement the same method name differently and calling code can work with any of them interchangeably without knowing the specific type. These four principles together enable modular reusable maintainable and extensible code.",
            "Define all four principles clearly with examples and explain why each matters in practice."
        ),

        (
            "What is the difference between == and is?",
            "In Python == and is are two distinct comparison operators that test fundamentally different things. The == operator compares the values of two objects checking whether the data or content stored in them is equal regardless of whether they are the same object in memory. Classes can override == behavior by defining the __eq__ method. The is operator compares object identity checking whether two variables refer to the exact same object at the same memory address not just objects with equal values. For example two lists created separately containing identical elements will return True for == because their values match but will return False for is because they are different objects occupying different memory locations. The most important practical rule is to always use is when comparing to None rather than ==. Because None is a singleton in Python there is exactly one None object and is None is both more correct and faster than equals equals None which could theoretically be overridden by a custom __eq__ method.",
            "Distinguish value equality from identity clearly explain the None case and give concrete examples."
        ),

    ],

    # ══════════════════════════════════════════════════════
    # SQL — DATABASE & QUERY CONCEPTS
    # ══════════════════════════════════════════════════════
    'sql': [

        (
            "What is the difference between INNER JOIN and LEFT JOIN?",
            "Both INNER JOIN and LEFT JOIN combine rows from two tables based on a matching condition but differ critically in what they return when no match exists. INNER JOIN returns only the rows where the join condition is satisfied in both tables. Any row in either table that has no matching counterpart in the other table is excluded entirely from the result set producing only the intersection of matched records. LEFT JOIN also called LEFT OUTER JOIN returns all rows from the left table regardless of whether a match exists in the right table. For left table rows with no matching right table row the columns from the right table appear as NULL values in the result. Use INNER JOIN when you only need records that exist in both tables such as orders with associated customers. Use LEFT JOIN when you need all records from the primary table and want to optionally include related data such as all customers including those who have never placed an order.",
            "Explain what each returns what happens with no match and give a real business scenario for each."
        ),

        (
            "What is a primary key?",
            "A primary key is a column or combination of columns in a relational database table that uniquely identifies each row. A primary key must satisfy three constraints: every value must be unique across all rows no value can be NULL and each table can have only one primary key. Primary keys are fundamental to relational databases because they enable other tables to reference specific rows through foreign keys forming the basis of relational data modeling and enabling JOIN operations. In PostgreSQL and most relational databases defining a primary key automatically creates a unique B-tree index on those columns which speeds up lookups by primary key value. Common choices include auto-incrementing integers using SERIAL for simplicity UUIDs for distributed systems where generating IDs without central coordination is needed and composite keys when uniqueness is only guaranteed by combining multiple columns such as a junction table in a many-to-many relationship.",
            "Define it cover all three constraints explain its role in foreign keys and discuss common implementation choices."
        ),

        (
            "Explain GROUP BY with HAVING.",
            "GROUP BY is a SQL clause that collapses multiple rows sharing the same value in specified columns into a single summary row allowing aggregate functions like COUNT SUM AVG MIN and MAX to be computed per group. For example SELECT department AVG of salary FROM employees GROUP BY department returns one summary row per department showing the average salary for that department. HAVING is a filter clause applied after grouping that allows filtering groups based on aggregate function results. It works like WHERE but WHERE filters individual rows before grouping while HAVING filters groups after aggregation. WHERE cannot reference aggregate functions but HAVING can. For example HAVING AVG of salary greater than 50000 retains only departments where the average salary exceeds 50000. The SQL execution order is FROM WHERE GROUP BY HAVING SELECT ORDER BY. Understanding this order is essential for writing logically correct queries.",
            "Define GROUP BY explain HAVING show the key difference from WHERE and state the SQL execution order."
        ),

        (
            "What are indexes and why are they used?",
            "An index in a relational database is a data structure that improves the speed of data retrieval by providing a fast lookup path to rows without requiring a full sequential scan of the entire table. Without an index the database engine must examine every row to find matches which has O of n time complexity and becomes unacceptably slow on large tables. With a B-tree index which is the default in PostgreSQL lookups have O of log n complexity. Indexes are most beneficial on columns frequently used in WHERE clauses JOIN conditions and ORDER BY expressions. Primary keys automatically receive an index and foreign key columns should also typically be indexed. The trade-off is that indexes consume additional disk storage and slow down INSERT UPDATE and DELETE operations because the index structure must be updated along with the table data. Over-indexing hurts write-heavy workloads. Index types in PostgreSQL include B-tree for equality and range queries Hash for equality only GIN for full-text search and array containment and BRIN for large sequentially ordered data like timestamps.",
            "Explain the performance benefit the trade-off index types and when to create or avoid indexes."
        ),

        (
            "Write a query to find the second highest salary.",
            "There are several correct approaches to finding the second highest salary in SQL. The subquery approach uses SELECT MAX of salary FROM employees WHERE salary is less than the result of SELECT MAX of salary FROM employees. The inner query finds the absolute highest salary and the outer query finds the maximum salary strictly below that value. The OFFSET approach uses SELECT DISTINCT salary FROM employees ORDER BY salary DESC LIMIT 1 OFFSET 1 which orders all unique salaries highest first and skips the top one to return the second. DISTINCT is important in both approaches to handle cases where multiple employees share the highest salary. For the general case of finding the Nth highest salary use a common table expression with DENSE_RANK: WITH ranked AS SELECT salary DENSE_RANK OVER ORDER BY salary DESC AS rank FROM employees then SELECT salary FROM ranked WHERE rank equals N. DENSE_RANK is preferred over RANK because it handles ties without creating gaps in the ranking sequence.",
            "Give multiple approaches explain why DISTINCT matters and show the general N-th highest using DENSE_RANK."
        ),

        (
            "What is the difference between DELETE, TRUNCATE, and DROP?",
            "DELETE TRUNCATE and DROP all remove data or structure but at fundamentally different levels with different behavior and consequences. DELETE is a DML statement that removes specific rows matching a WHERE condition. It is logged row by row can be rolled back within a transaction fires ON DELETE triggers and is slow on large tables because each deleted row is individually processed. Without a WHERE clause it deletes all rows but slowly. TRUNCATE is a DDL statement that removes all rows from a table instantly by deallocating the entire data pages rather than processing rows individually. It is dramatically faster than DELETE for clearing whole tables resets identity counters in most databases cannot be filtered with WHERE and in PostgreSQL can actually be rolled back unlike in MySQL. DROP removes the entire table definition including all its data indexes constraints and triggers permanently. The table completely ceases to exist. This is irreversible without a backup. Choose DELETE for selective row removal TRUNCATE for fast complete table clearing and DROP only when the table structure itself is no longer needed.",
            "Distinguish all three on speed rollback behavior triggers row selection and appropriate use case."
        ),

        (
            "What is a foreign key?",
            "A foreign key is a column or set of columns in one table called the child or referencing table that references the primary key of another table called the parent or referenced table creating a formal enforced relationship between them. Foreign keys enforce referential integrity meaning the database will automatically reject any INSERT or UPDATE that would create a foreign key value with no corresponding primary key value in the parent table preventing orphaned child records. Foreign keys also define referential actions for when a parent row is deleted or updated through ON DELETE and ON UPDATE clauses. CASCADE propagates the change to all matching child rows. SET NULL sets the foreign key column to NULL. RESTRICT prevents the DELETE or UPDATE operation if matching child rows exist. NO ACTION is similar to RESTRICT but deferred. Foreign keys make the database engine responsible for maintaining relationship consistency removing that burden from application code and preventing data corruption.",
            "Define it explain referential integrity cover all referential actions and state why it matters over application-level checks."
        ),

        (
            "What is normalization?",
            "Normalization is the systematic process of organizing a relational database schema to reduce data redundancy eliminate data anomalies and improve data integrity by decomposing tables according to a series of normal forms. First Normal Form requires that each column hold atomic indivisible single values with no repeating groups or arrays within a cell and that each row be uniquely identifiable. Second Normal Form requires the table be in 1NF and that every non-key column be fully functionally dependent on the entire primary key not just a part of it which specifically addresses tables with composite primary keys. Third Normal Form requires the table be in 2NF and that no non-key column depend on another non-key column eliminating transitive dependencies. Normalization prevents update anomalies where changing one value requires updating many rows insertion anomalies where you cannot add a row without unrelated required data and deletion anomalies where deleting one record inadvertently removes other needed information. Production systems sometimes deliberately denormalize hot read paths for query performance accepting controlled redundancy in exchange for fewer joins.",
            "Define the purpose explain 1NF through 3NF with what each eliminates and mention denormalization trade-offs."
        ),

        (
            "What is the difference between UNION and UNION ALL?",
            "UNION and UNION ALL both combine the result sets of two or more SELECT queries into a single result but differ in how they handle duplicate rows. UNION removes duplicate rows from the combined result performing an implicit DISTINCT operation after merging both result sets. This deduplication requires an additional sort or hash operation making UNION slower than UNION ALL especially on large result sets. UNION ALL includes every row from all queries including duplicates without any deduplication step making it faster because it avoids the extra processing overhead. For both to work all combined SELECT statements must have the same number of columns and the data types in corresponding column positions must be compatible. Use UNION when correctness requires each unique row to appear only once such as merging customer lists from two systems where the same customer might exist in both. Use UNION ALL when you know duplicates cannot occur or when you intentionally want all occurrences such as computing total event counts across multiple partitioned tables.",
            "Explain deduplication the performance difference compatibility requirements and give a real scenario for each."
        ),

        (
            "Explain ACID properties.",
            "ACID is an acronym for four properties that guarantee reliable and consistent processing of database transactions even in the presence of system failures or concurrent access. Atomicity means a transaction is treated as a single indivisible unit of work. Either all statements within it succeed and are committed or if any statement fails the entire transaction is rolled back leaving the database unchanged as if the transaction never started. Consistency means a transaction always transitions the database from one valid state to another valid state respecting all defined constraints rules and data integrity requirements. A transaction that would violate a constraint is rejected. Isolation means concurrent transactions execute without interfering with each other. The intermediate uncommitted state of one transaction is invisible to other transactions. Isolation levels from weakest to strongest are READ UNCOMMITTED READ COMMITTED REPEATABLE READ and SERIALIZABLE each trading more isolation for less concurrency throughput. Durability means once a transaction is committed its changes are permanently recorded and survive system crashes power failures or other failures. This is typically achieved through write-ahead logging where changes are recorded to a durable log before being applied to data pages.",
            "Define all four properties precisely explain isolation levels and describe how durability is achieved."
        ),

    ],

    # ══════════════════════════════════════════════════════
    # ML — MACHINE LEARNING THEORY & ALGORITHMS
    # ══════════════════════════════════════════════════════
    'ml': [

        (
            "Explain overfitting and underfitting.",
            "Overfitting occurs when a machine learning model learns the training data too precisely capturing noise and random fluctuations along with the genuine patterns. An overfit model achieves very low training error but high test error because it has essentially memorized the training examples rather than learned generalizable patterns. Underfitting occurs when a model is too simple to capture the true underlying structure in the data producing high error on both training and test sets. The bias-variance tradeoff precisely frames this: overfitting corresponds to high variance meaning small changes in training data cause large changes in the model while underfitting corresponds to high bias meaning the model consistently misses the true signal. Techniques to combat overfitting include L1 and L2 regularization dropout in neural networks early stopping cross-validation reducing model complexity and acquiring more diverse training data. Techniques to combat underfitting include increasing model capacity adding relevant features and training for more iterations. The goal is the sweet spot where the model captures the true signal without memorizing the noise.",
            "Define both connect to bias-variance give practical remedies for each and state the overall goal."
        ),

        (
            "What is gradient descent?",
            "Gradient descent is an iterative first-order optimization algorithm used to minimize the loss function of a machine learning model by repeatedly updating the model's parameters. The algorithm computes the gradient of the loss function with respect to each parameter which points in the direction of steepest increase in loss and then moves each parameter a small step in the exact opposite direction to reduce the loss. The size of each step is controlled by the learning rate hyperparameter. Too large a learning rate causes the algorithm to overshoot the minimum and diverge. Too small a learning rate makes convergence extremely slow requiring many more iterations. Batch gradient descent uses the full training dataset to compute the gradient for each update which is accurate but very slow on large datasets. Stochastic gradient descent uses a single random sample per update which is fast and noisy but can escape local minima. Mini-batch gradient descent uses a small batch of samples typically 32 to 256 and is the most widely used variant balancing computational efficiency with stability. Modern adaptive optimizers like Adam RMSprop and AdaGrad extend gradient descent by maintaining per-parameter adaptive learning rates using the history of past gradients achieving faster and more robust convergence.",
            "Define the update rule explain learning rate effects compare SGD variants and mention adaptive optimizers."
        ),

        (
            "What is the difference between supervised and unsupervised learning?",
            "Supervised learning trains a model on a labelled dataset where each training example consists of an input and a corresponding target output. The model learns a function mapping inputs to outputs and is evaluated on how accurately it predicts the correct output for new unseen inputs. Classification predicting a discrete class label and regression predicting a continuous numerical value are the two main supervised learning task types. Real examples include email spam detection house price prediction and medical diagnosis. Unsupervised learning trains on data with no labels. The model finds hidden structure patterns or groupings in the data without any predefined target to predict. Clustering algorithms group similar data points together dimensionality reduction compresses high-dimensional data while preserving structure and anomaly detection identifies unusual patterns. Real examples include customer segmentation topic modeling and fraud detection in unlabelled transaction streams. Semi-supervised learning uses a small amount of labelled data combined with a large amount of unlabelled data which reflects many real-world scenarios where labelling is expensive and slow but raw data is abundant.",
            "Define both give task subtypes with real examples for each and explain semi-supervised as a practical middle ground."
        ),

        (
            "What is regularization?",
            "Regularization is a collection of techniques that prevent overfitting by adding constraints or penalties that discourage the model from learning overly complex patterns that do not generalize. The two most common mathematical forms are L1 and L2 regularization. L1 regularization called Lasso adds the sum of the absolute values of all model weights to the loss function. This penalty has the distinctive effect of driving many weights exactly to zero during optimization effectively performing automatic feature selection by eliminating irrelevant features from the model entirely. L2 regularization called Ridge adds the sum of the squared values of all model weights to the loss. This shrinks all weights toward zero proportionally but rarely eliminates them completely producing a simpler model with more evenly distributed small weights. Elastic Net combines both L1 and L2 penalties capturing the benefits of both. The regularization strength is controlled by a hyperparameter lambda or alpha: larger values apply stronger regularization creating simpler models at the cost of increased bias. In neural networks dropout regularization randomly deactivates a fraction of neurons during each training step preventing neurons from co-adapting and forcing the network to learn more robust redundant representations.",
            "Define the purpose explain L1 vs L2 including feature selection difference cover Elastic Net and dropout."
        ),

        (
            "Explain the bias-variance tradeoff.",
            "The bias-variance tradeoff is a fundamental decomposition of prediction error in machine learning that explains why models make mistakes and why improving one type of error often worsens the other. Bias is systematic error introduced by overly simplistic assumptions in the model. A high-bias model consistently misses the true relationship between inputs and outputs performing poorly even on training data and causing underfitting. Variance is the error from the model's sensitivity to small random fluctuations in the training data. A high-variance model is too complex fitting the training data extremely well including its noise but producing wildly different results on slightly different training sets causing overfitting. As model complexity increases bias decreases because the model can capture more complex patterns but variance increases because it also starts fitting noise. As model complexity decreases bias increases and variance decreases. The total expected prediction error of a model equals bias squared plus variance plus irreducible noise which is the minimum possible error from factors outside the model's control. The practical goal is to find the model complexity that minimizes total error and cross-validation is the primary tool for estimating this tradeoff empirically on real data.",
            "Define both components connect to overfitting and underfitting state the error decomposition and give the practical solution."
        ),

        (
            "What is cross-validation?",
            "Cross-validation is a statistical technique for estimating how well a machine learning model will generalize to independent data by systematically testing it on different held-out subsets of the available data. The most common form is k-fold cross-validation. The dataset is divided into k equal-sized non-overlapping folds. The model is trained on k minus one folds and evaluated on the remaining held-out fold. This process is repeated k times using each fold exactly once as the validation set. The final performance estimate is the average score across all k runs which is more reliable than any single train-test split. Typical values of k are 5 and 10 balancing reliability with computational cost. Stratified k-fold is a critical variant for classification tasks that ensures each fold contains approximately the same proportion of each class as the complete dataset which prevents misleading results on imbalanced datasets. Cross-validation also serves as the foundation for hyperparameter tuning through grid search or random search where each candidate configuration is evaluated using cross-validation to compare configurations fairly on the same data folds. Leave-one-out cross-validation where k equals the total number of samples is maximally data efficient but computationally prohibitive for large datasets.",
            "Describe the k-fold procedure step by step explain why it beats a single split and cover stratified k-fold and hyperparameter tuning."
        ),

        (
            "What is a confusion matrix?",
            "A confusion matrix is a tabular summary of a classification model's prediction performance that breaks down results by actual versus predicted class labels providing far more diagnostic information than a single accuracy number. For binary classification it is a 2 by 2 matrix with four cells. True Positives are cases where the model correctly predicted positive and the actual label was positive. True Negatives are cases where the model correctly predicted negative and the actual label was negative. False Positives are cases where the model incorrectly predicted positive when the actual label was negative representing a Type I error or false alarm. False Negatives are cases where the model incorrectly predicted negative when the actual label was positive representing a Type II error or missed detection. From these four values all major evaluation metrics are derived. Accuracy is TP plus TN divided by the total number of predictions. Precision is TP divided by TP plus FP measuring how often positive predictions are correct. Recall also called sensitivity is TP divided by TP plus FN measuring how many actual positives were captured. F1-score is the harmonic mean of precision and recall providing a single balanced metric. For multi-class problems the matrix extends to N by N where correct predictions appear on the diagonal and off-diagonal entries reveal which classes the model confuses with each other.",
            "Define all four cells explain Type I and II errors derive all key metrics and describe the multi-class extension."
        ),

        (
            "What is the difference between precision and recall?",
            "Precision and recall are complementary evaluation metrics for classification models that measure different and sometimes competing aspects of prediction quality and are especially important when class distributions are imbalanced. Precision measures exactness: of all instances the model predicted as positive what fraction were actually positive. High precision means the model rarely raises false alarms. The formula is True Positives divided by True Positives plus False Positives. Recall also called sensitivity measures completeness: of all actual positive instances in the dataset what fraction did the model correctly identify. High recall means the model misses few real positives. The formula is True Positives divided by True Positives plus False Negatives. There is a fundamental trade-off between the two. Lowering the classification threshold makes the model more aggressive in predicting positive increasing recall but also increasing false positives and reducing precision. Raising the threshold makes predictions more conservative increasing precision but missing more true positives and reducing recall. F1-score is the harmonic mean of precision and recall providing a single metric that balances both. The right balance depends entirely on the cost of each error type in the specific application domain. In cancer screening high recall is critical because missing a real cancer is far more harmful than an unnecessary follow-up test. In email spam filtering high precision may matter more because sending legitimate emails to spam is more damaging than letting occasional spam through.",
            "Define both with formulas explain the trade-off the threshold effect and give domain-specific examples of when each matters more."
        ),

        (
            "What is a random forest?",
            "Random Forest is an ensemble learning algorithm that constructs a large collection of decision trees during training and aggregates their individual predictions to produce a final more accurate and robust result. The algorithm introduces two key sources of randomness to ensure the trees are diverse and not correlated with each other. First each tree is trained on a different random bootstrap sample of the training data drawn with replacement a technique called bagging or bootstrap aggregating. Some training examples appear multiple times in a tree's sample while others may not appear at all. Second at each node split in every tree only a random subset of features is considered as candidates for splitting rather than all available features. This feature randomness prevents all trees from learning the same dominant split patterns and ensures genuine diversity across the ensemble. For classification the final prediction is the majority vote across all trees. For regression it is the mean prediction. Because the averaging process cancels out individual errors Random Forest achieves significantly lower variance than any single decision tree while maintaining similar bias making it highly resistant to overfitting. It naturally produces feature importance scores based on how much each feature contributes to reducing impurity across all trees. Key hyperparameters are the number of trees maximum depth per tree and the number of features considered at each split.",
            "Explain bagging feature randomness prediction aggregation the variance reduction benefit and feature importance."
        ),

        (
            "Explain DBSCAN clustering.",
            "DBSCAN standing for Density-Based Spatial Clustering of Applications with Noise is a clustering algorithm that groups data points based on local density rather than distance to a centroid making it fundamentally different from centroid-based methods like K-means. It requires two hyperparameters: epsilon which defines the radius of the local neighborhood around each point and min_samples which is the minimum number of points required within that epsilon radius for a point to qualify as a core point. A core point has at least min_samples neighbors within its epsilon radius and forms the nucleus of a cluster. A border point falls within the epsilon radius of a core point but does not itself have enough neighbors to be a core point. It belongs to the cluster of the core point nearest to it. A noise point is not reachable from any core point and is classified as an outlier assigned the label negative one. Clusters are formed by connecting all core points that are within epsilon of each other and including their reachable border points forming arbitrarily shaped connected dense regions. The critical advantages over K-means are that DBSCAN does not require specifying the number of clusters in advance it can discover clusters of completely arbitrary non-convex shapes and it explicitly identifies and isolates noise points as outliers rather than forcing them into the nearest cluster. It is ideal for geospatial data anomaly detection and applications where clusters have irregular shapes.",
            "Explain both parameters all three point types cluster formation rules and specific advantages over K-means."
        ),

    ],

    # ══════════════════════════════════════════════════════
    # DATA SCIENCE — STATISTICS, EDA & METHODOLOGY
    # ══════════════════════════════════════════════════════
    'ds': [

        (
            "What is EDA?",
            "Exploratory Data Analysis or EDA is the mandatory first phase of any data science project where the analyst systematically examines the dataset to understand its structure quality distributions and relationships before building any model or drawing any conclusions. EDA serves two primary purposes: identifying data quality problems such as missing values outliers inconsistent formatting duplicate records and incorrect data types and developing deep understanding of the patterns correlations and distributions that will inform feature engineering and model selection decisions. Typical EDA steps include examining the shape and data types of the dataset computing summary statistics like mean median standard deviation minimum maximum and quantiles for numerical columns examining value counts for categorical columns visualizing distributions with histograms box plots and violin plots detecting correlations between features with heatmaps identifying outliers with scatter plots and z-scores and checking the extent and pattern of missing values. A well-executed EDA often reveals that the data itself requires significant cleaning transformation or enrichment before any modeling will be meaningful. Skipping EDA is one of the most common mistakes in data science and frequently leads to models built on dirty or misunderstood data.",
            "Define EDA state its two primary purposes list the concrete steps and explain the consequences of skipping it."
        ),

        (
            "How do you handle missing values?",
            "Handling missing values correctly is one of the most consequential preprocessing decisions in data science because the wrong strategy can introduce systematic bias destroy valuable signal or leak information from test to training data. The first and most important step is diagnosing why data is missing. Missing completely at random means missingness has no pattern and any handling strategy is acceptable. Missing at random means missingness correlates with other observed variables and must be accounted for. Missing not at random means the missing value itself determines whether it is missing which is the most dangerous case requiring careful domain judgment. For numerical features common strategies include mean imputation for symmetric distributions median imputation for skewed distributions and model-based imputation using KNN imputer or iterative imputer for more accurate estimates preserving relationships between variables. For categorical features imputation with the mode or creating an explicit Unknown category are appropriate. Columns with more than 50 to 60 percent missing values should usually be dropped entirely. For time series forward fill and backward fill respect temporal order. The golden rule is to fit all imputation statistics exclusively on training data and apply those same statistics to validation and test data to prevent data leakage which would give optimistically biased performance estimates.",
            "Explain the three types of missingness give strategies for numerical and categorical features and emphasize data leakage prevention."
        ),

        (
            "What is feature engineering?",
            "Feature engineering is the process of using domain knowledge and analytical insight to create transform combine or select input features from raw data in ways that make machine learning models significantly more effective than they would be on the raw data alone. Raw data is rarely in the ideal format for a learning algorithm and feature engineering bridges that gap. Common encoding techniques include one-hot encoding for nominal categorical variables and ordinal encoding for ordered categories. Numerical features often benefit from standardization or min-max scaling to prevent features with larger ranges from dominating distance-based algorithms. Creating interaction features by multiplying or combining existing variables captures relationships the model cannot discover independently. Datetime columns are rich sources of features including hour of day day of week month quarter and indicators for holidays or seasons. Log transformation and Box-Cox transformation handle right-skewed distributions making them more normally distributed which benefits many algorithms. Binning continuous variables into categorical buckets can capture nonlinear threshold effects. Domain-specific features like transaction velocity for fraud detection session depth for user engagement or word count and sentence complexity for text analysis often provide the largest model improvements. Feature selection through techniques like correlation analysis recursive feature elimination or model-based importance scores removes irrelevant noisy features that would otherwise hurt generalization.",
            "Define it list specific technique categories with concrete examples and explain why features often matter more than the algorithm."
        ),

        (
            "What is the difference between correlation and causation?",
            "Correlation describes a statistical relationship between two variables where changes in one tend to systematically accompany changes in the other. Positive correlation means both variables increase together. Negative correlation means one increases as the other decreases. Correlation is quantified by coefficients such as Pearson r for linear relationships or Spearman rho for monotonic relationships both ranging from negative one to positive one. Causation means that one variable directly produces changes in another through a real mechanism not merely that they tend to move together in observed data. The critical principle is that correlation does not imply causation. Two variables can be correlated for three distinct reasons: X directly causes Y Y directly causes X or a third confounding variable Z independently causes both X and Y creating a spurious apparent correlation between them with no direct causal link. A famous illustrative example is the strong positive correlation between ice cream sales and drowning deaths. Both are caused by hot summer weather a confounder and ice cream has absolutely no causal effect on drowning risk. In data science predictive models can legitimately exploit correlations to make accurate predictions without understanding causation but informing business decisions and policy requires causal understanding. Establishing causation requires randomized controlled experiments where the confounding variables are held constant or eliminated through randomization or advanced causal inference methods like instrumental variables regression discontinuity or difference-in-differences.",
            "Define both with formulas explain three reasons for correlation give a concrete example and state how causation is established."
        ),

        (
            "Explain the Central Limit Theorem.",
            "The Central Limit Theorem is one of the most powerful and practically important results in probability theory and mathematical statistics. It states that the sampling distribution of the sample mean converges to a normal distribution as the sample size n increases regardless of the shape of the original population distribution provided the individual samples are independent and identically distributed and the population has a finite mean and finite variance. This result is remarkable precisely because the population itself does not need to be normally distributed at all. Whether the population follows a skewed distribution a uniform distribution an exponential distribution or any other shape the distribution of the average across repeated samples of size n will become increasingly bell-shaped as n grows larger. In practice the normal approximation is generally adequate for sample sizes of 30 or more for most population shapes though more highly skewed or heavy-tailed distributions may require larger samples. The sampling distribution has a mean equal to the population mean and a standard deviation called the standard error equal to the population standard deviation divided by the square root of n. As n increases the standard error shrinks meaning larger samples produce more precise estimates of the population mean. The Central Limit Theorem provides the theoretical justification for nearly all parametric hypothesis tests and confidence interval procedures including the t-test and z-test explaining why these methods remain valid and widely used even when the underlying data is not normally distributed.",
            "State the theorem precisely explain why it is remarkable state the standard error formula and explain its practical importance for hypothesis testing."
        ),

        (
            "What is a p-value?",
            "A p-value is the probability of observing a test statistic at least as extreme as the value computed from the sample data assuming the null hypothesis is true and the study were repeated under identical conditions many times. It quantifies how surprising the observed data is under the assumption that nothing interesting is happening. A small p-value indicates that the observed result would be very unlikely if the null hypothesis were true providing statistical evidence to reject it. The significance threshold alpha is set in advance before looking at data and is most commonly 0.05 representing a 5 percent accepted risk of falsely rejecting a true null hypothesis which is a Type I error. If the p-value is less than alpha we reject the null hypothesis. If it is greater than alpha we fail to reject it. Failing to reject does not mean the null hypothesis is true it simply means the data does not provide sufficient evidence to reject it. There are several critical misconceptions about p-values that practitioners must avoid. The p-value is not the probability that the null hypothesis is true. It is not the probability the result was due to chance. It is a conditional probability given that the null hypothesis is true. P-values are also highly sensitive to sample size: with very large samples even trivially small and practically meaningless differences become statistically significant. This is why statistical significance must always be interpreted alongside effect size measures like Cohen's d or correlation coefficients which quantify the practical magnitude of the observed difference.",
            "Define it precisely explain the significance threshold state what failing to reject means give common misconceptions and explain the sample size sensitivity."
        ),

        (
            "What are Type I and Type II errors?",
            "Type I and Type II errors are the two fundamental mistakes a statistical hypothesis test can make and understanding their trade-off is essential for designing experiments and interpreting results correctly. A Type I error also called a false positive occurs when the test incorrectly rejects a null hypothesis that is actually true. You conclude an effect exists when it does not. The probability of committing a Type I error is controlled by the significance level alpha which is set before testing. Setting alpha to 0.05 means you accept a 5 percent chance of falsely detecting an effect that is not real. A Type II error also called a false negative or missed detection occurs when the test fails to reject a null hypothesis that is actually false. You miss a real effect that genuinely exists in the population. The probability of a Type II error is called beta and statistical power defined as one minus beta is the probability of correctly detecting a real effect when one is present. There is a direct and unavoidable trade-off between the two types of errors. Making alpha smaller to reduce false positives increases beta and reduces power making it harder to detect real effects. The correct balance depends entirely on the relative costs of each error in the specific domain. In cancer screening a Type II error meaning missing a real cancer is catastrophic and life-threatening so tests are designed with very high sensitivity and high power accepting more false positives for the benefit of catching nearly every real case. In criminal justice a Type I error meaning convicting an innocent person is considered the graver injustice so the standard of proof is set extremely high to minimize false positives even at the cost of some false negatives.",
            "Define both errors connect to alpha and power explain the trade-off and give domain-specific examples showing which error is costlier."
        ),

        (
            "What is dimensionality reduction?",
            "Dimensionality reduction is the process of reducing the number of features in a dataset while preserving as much of the meaningful information and structure as possible. High-dimensional data suffers from the curse of dimensionality: as the number of dimensions increases data becomes exponentially sparse distances between points lose discriminative power and models require exponentially more data to generalize well. Many real-world datasets also contain redundant or correlated features that carry duplicate information. Principal Component Analysis or PCA is the most widely used linear dimensionality reduction technique. It identifies orthogonal directions called principal components that explain the maximum variance in the data and projects the data onto a lower-dimensional subspace defined by the top k components ranked by explained variance. PCA is computationally efficient preserves global structure and is invertible. t-SNE which stands for t-distributed Stochastic Neighbor Embedding is a nonlinear technique used primarily for 2D or 3D visualization of high-dimensional data. It excels at preserving local neighborhood structure revealing clusters and patterns invisible in the original space but distorts global distances and should not be used as a preprocessing step before modeling. UMAP is a newer nonlinear technique that is significantly faster than t-SNE scales better to large datasets and better preserves both local and global structure. Autoencoders are neural network-based dimensionality reduction methods capable of learning highly nonlinear compressed representations. Applications of dimensionality reduction include speeding up downstream model training reducing overfitting noise filtering enabling visualization and compressing data for efficient storage.",
            "Explain the curse of dimensionality describe PCA t-SNE UMAP and autoencoders with their specific strengths and give applications."
        ),

        (
            "What is A/B testing?",
            "A/B testing is a controlled randomized experiment that compares two versions of something typically a product feature user interface design pricing strategy algorithm or marketing message to determine which version performs better on a predefined success metric. Users or subjects are randomly assigned to either the control group A which experiences the current baseline version or the treatment group B which experiences the new version. Random assignment is the most critical element because it ensures the two groups are statistically comparable controlling for all confounding variables and allowing any observed difference in outcomes to be causally attributed to the tested change rather than to pre-existing differences between the groups. After collecting data for a predetermined period statistical hypothesis testing is applied to determine whether the observed difference in the success metric between groups is statistically significant or within the range expected from random variation. Critical design considerations include calculating the required sample size in advance based on desired statistical power and minimum detectable effect to avoid underpowered tests that cannot detect real differences. Researchers must avoid peeking at results before the planned sample size is reached because early stopping dramatically inflates the false positive rate. Novelty effects where users engage more with any new change regardless of its quality must be accounted for by running experiments long enough for behavior to stabilize. The measured metric must genuinely reflect the true business objective not just a superficial proxy. A/B testing is the gold standard for data-driven product decisions because it provides causal evidence rather than mere correlation.",
            "Explain the setup why randomization is essential statistical testing design pitfalls and why it provides causal evidence."
        ),

        (
            "What is the difference between a bar chart and a histogram?",
            "Bar charts and histograms are visually similar but represent fundamentally different types of data and answer completely different analytical questions requiring careful attention to which is appropriate. A bar chart is used to display and compare categorical or discrete data. Each bar represents a distinct named category and its height encodes the value or frequency for that category. The bars are visually separated by gaps reinforcing that the categories are discrete independent entities with no natural ordering or numerical relationship between adjacent bars. The order of bars can be rearranged without changing the meaning of the chart. A histogram is used to display the distribution of a single continuous numerical variable. The entire data range is divided into contiguous adjacent intervals called bins and each bar's height represents the count or probability density of observations falling within that bin interval. The bars touch with no gaps because the underlying variable is continuous with no natural breaks between values. The choice of bin width has a significant impact on the shape revealed: too few bins overly smooth the distribution hiding important features while too many bins create noise making patterns hard to see. Choosing bar count using rules like Sturges' rule or the Freedman-Diaconis estimator or using kernel density estimation as an alternative helps select appropriate binning. Using the wrong chart type for your data type produces misleading visualizations. Use a bar chart to compare discrete categories. Use a histogram to understand the shape center spread skewness and modality of a continuous variable.",
            "Explain data type suitability visual difference the significance of gaps bin choice impact and the analytical purpose of each."
        ),

    ],
}
def seed_questions():
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute('SELECT COUNT(*) FROM questions')
            if cur.fetchone()[0] > 0:
                return
            for cat, items in SEED_DATA.items():
                for q_text, ideal, key in items:
                    cur.execute(
                        '''INSERT INTO questions (category, question_text, ideal_answer, key, difficulty)
                           VALUES (%s, %s, %s, %s, %s)''',
                        (cat, q_text, ideal, key, 'medium')
                    )
        conn.commit()


init_db()

def get_account_by_email(email):
    with get_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                'SELECT id, first_name, last_name, email, password FROM accounts WHERE email = %s',
                (email.lower().strip(),)
            )
            return cur.fetchone()


def create_account(first_name, last_name, email, password):
    hashed = generate_password_hash(password)
    with get_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                'INSERT INTO accounts (first_name, last_name, email, password) VALUES (%s,%s,%s,%s) RETURNING id',
                (first_name, last_name, email.lower().strip(), hashed)
            )
            user_id = cur.fetchone()['id']
            # Mirror into users table
            try:
                cur.execute(
                    'INSERT INTO users (name, email) VALUES (%s,%s) ON CONFLICT (email) DO NOTHING',
                    (first_name, email.lower().strip())
                )
            except Exception:
                pass
        conn.commit()
    return user_id


def get_user_id_from_users_table(email):
    with get_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute('SELECT id FROM users WHERE email = %s', (email,))
            row = cur.fetchone()
            return row['id'] if row else None

def score_answer(user_answer, ideal_answer):
    if not user_answer or not ideal_answer:
        return 0.0
    user_embedding = nlp_model.encode(user_answer)
    ideal_embedding = nlp_model.encode(ideal_answer)
    similarity = cosine_similarity([user_embedding], [ideal_embedding])[0][0]
    return float(round(float(similarity) * 100, 1))


def get_user_context():
    first_name = session.get('user_first_name', '')
    email      = session.get('user_email', '')
    initial    = first_name[0].upper() if first_name else 'G'
    current_user = SimpleNamespace(
        name=first_name or 'Guest',
        email=email or '',
        initial=initial,
    )
    return dict(
        user_name=first_name or 'Guest',
        user_email=email or '',
        user_initial=initial,
        current_user=current_user,
    )


@app.context_processor
def inject_user():
    return {'current_user': get_user_context()['current_user']}


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_email' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_email' in session:
        return redirect(url_for('dashboard'))
    error = None
    if request.method == 'POST':
        email    = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        account  = get_account_by_email(email)
        if account and check_password_hash(account['password'], password):
            session['user_id']         = account['id']
            session['user_email']      = account['email']
            session['user_first_name'] = account['first_name'] or ''
            return redirect(url_for('dashboard'))
        error = 'Invalid email or password.'
    return render_template('login.html', error_message=error)


@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'user_email' in session:
        return redirect(url_for('dashboard'))
    error = None
    if request.method == 'POST':
        first_name = request.form.get('first_name', '').strip()
        last_name  = request.form.get('last_name', '').strip()
        email      = request.form.get('email', '').strip().lower()
        password   = request.form.get('password', '')
        confirm    = request.form.get('confirm_password', '')
        if not first_name or not email or not password:
            error = 'Please fill in all required fields.'
        elif password != confirm:
            error = 'Passwords do not match.'
        elif get_account_by_email(email):
            error = 'Email already registered. Please log in.'
        else:
            uid = create_account(first_name, last_name, email, password)
            session['user_id']         = uid
            session['user_email']      = email
            session['user_first_name'] = first_name
            return redirect(url_for('dashboard'))
    return render_template('register.html', error_message=error)


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))



@app.route('/categories')
@login_required
def categories():
    return render_template('categories.html', **get_user_context())


@app.route('/dashboard')
@login_required
def dashboard():
    users_id = get_user_id_from_users_table(session['user_email'])

    recent_sessions = []
    total_interviews = 0
    avg_score  = 0.0
    best_score = 0.0
    cat_scores = []
    chart_labels = []
    chart_scores_list = []

    if users_id:
        with get_db() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:

                cur.execute('''
                    SELECT category, overall_score, total_questions, interview_date
                    FROM interview_session WHERE user_id = %s
                    ORDER BY interview_date DESC LIMIT 10
                ''', (users_id,))
                recent_sessions = cur.fetchall()

                cur.execute('''
                    SELECT COUNT(*) as total,
                           COALESCE(AVG(overall_score),0) as avg,
                           COALESCE(MAX(overall_score),0) as best
                    FROM interview_session WHERE user_id = %s
                ''', (users_id,))
                stats = cur.fetchone()
                total_interviews = stats['total']
                avg_score  = round(stats['avg'],  1)
                best_score = round(stats['best'], 1)

                cur.execute('''
                    SELECT category, ROUND(AVG(overall_score)::numeric,1) as avg
                    FROM interview_session WHERE user_id = %s
                    GROUP BY category ORDER BY avg DESC
                ''', (users_id,))
                cat_scores = cur.fetchall()

                cur.execute('''
                    SELECT overall_score, interview_date
                    FROM interview_session WHERE user_id = %s
                    ORDER BY interview_date ASC LIMIT 20
                ''', (users_id,))
                chart_rows = cur.fetchall()
                chart_labels      = [r['interview_date'].strftime('%b %d') for r in chart_rows]
                chart_scores_list = [round(r['overall_score'], 1) for r in chart_rows]

    return render_template('dashboard.html',
        **get_user_context(),
        recent_sessions=recent_sessions,
        total_interviews=total_interviews,
        avg_score=avg_score,
        best_score=best_score,
        cat_scores=cat_scores,
        chart_labels=chart_labels,
        chart_scores=chart_scores_list,
    )




@app.route('/interview/start')
@login_required
def interview_start():
    category  = request.args.get('category', 'hr').lower()

    with get_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                '''SELECT id, question_text, ideal_answer, hint
                   FROM questions WHERE category = %s
                   ORDER BY id LIMIT 10''',
                (category,)
            )
            questions = cur.fetchall()

    if not questions:
        return redirect(url_for('categories'))

    # Store in Flask session
    session['interview_category']      = category
    session['interview_questions']     = [dict(q) for q in questions]
    session['interview_question_index']= 0
    session['interview_answers']       = []
    session['interview_session_id']    = None
    session['interview_complete']      = False

    # Create interview_session row in DB (uses users table user_id)
    users_id = get_user_id_from_users_table(session['user_email'])
    if users_id:
        with get_db() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    '''INSERT INTO interview_session (user_id, category, overall_score, total_questions)
                       VALUES (%s, %s, %s, %s) RETURNING id''',
                    (users_id, category, 0.0, len(questions))
                )
                session['interview_session_id'] = cur.fetchone()['id']
            conn.commit()
            from sys import getsizeof

            print("SESSION CONTENTS")
            for k, v in session.items():
                print(f"{k}: {type(v)}")

            print("Questions stored:", len(session["interview_questions"]))
            print("Approx size:", getsizeof(str(session["interview_questions"])))
    
    return redirect(url_for('interview'))


@app.route('/interview')
@login_required
def interview():
    if not session.get('interview_category'):
        return redirect(url_for('categories'))

    questions     = session.get('interview_questions', [])
    current_index = session.get('interview_question_index', 0)
  
    if current_index >= len(questions):
        return redirect(url_for('output'))

    cat_labels = {
        'hr': 'HR Interview', 'python': 'Python Interview',
        'sql': 'SQL Interview', 'ml': 'Machine Learning', 'ds': 'Data Science',
    }

    return render_template('interview.html',
        questions=questions,
        current_index=current_index,
        total=len(questions),
        progress=round((current_index / len(questions)) * 100),
        category_label=cat_labels.get(session['interview_category'], 'Interview'),
        category=session['interview_category'],
    )


@app.route('/interview/answer', methods=['POST'])

@login_required
def interview_answer():
    if not session.get('interview_category'):
        return redirect(url_for('categories'))

    answer      = request.form.get('answer', '').strip()
    question_id = request.form.get('question_id', '')
    action      = request.form.get('action', 'next')

    questions     = session.get('interview_questions', [])
    current_index = session.get('interview_question_index', 0)

    # Score the answer
    current_q = questions[current_index] if current_index < len(questions) else {}
    sim_score = score_answer(answer, current_q.get('ideal_answer', ''))

    # Accumulate answers
    answers = session.get('interview_answers', [])
    answers.append({
        'question_id':   question_id,
        'question_text': current_q.get('question_text', ''),
        'user_answer':   answer,
        'score':         sim_score,
        'ideal_answer':  current_q.get('ideal_answer', ''),
    })
    session['interview_answers'] = answers

    # Save answer to DB
    users_id   = get_user_id_from_users_table(session['user_email'])
    session_id = session.get('interview_session_id')
    if users_id and session_id and question_id:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    '''INSERT INTO user_answers (session_id, user_id, question_id, user_answer, score)
                       VALUES (%s,%s,%s,%s,%s)''',
                    (session_id, users_id, question_id, answer, sim_score)
                )
            conn.commit()

    next_index = current_index + 1
    session['interview_question_index'] = next_index

    # Last question or user clicked Submit
    if action == 'submit' or next_index >= len(questions):
        all_scores = [a['score'] for a in answers]
        overall    = round(sum(all_scores) / len(all_scores), 1) if all_scores else 0.0
        session['interview_score']    = overall
        session['interview_complete'] = True

        # Update interview_session overall score
        if session_id:
            with get_db() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        'UPDATE interview_session SET overall_score=%s WHERE id=%s',
                        (overall, session_id)
                    )
                conn.commit()

        # Upsert performance table
        if users_id:
            cat = session['interview_category']
            with get_db() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        'SELECT performance_id, average_score FROM performance WHERE id=%s AND category=%s',
                        (users_id, cat)
                    )
                    existing = cur.fetchone()
                    if existing:
                        new_avg = round((existing['average_score'] + overall) / 2, 1)
                        cur.execute(
                            'UPDATE performance SET average_score=%s WHERE performance_id=%s',
                            (new_avg, existing['performance_id'])
                        )
                    else:
                        cur.execute(
                            'INSERT INTO performance (id, category, average_score) VALUES (%s,%s,%s)',
                            (users_id, cat, overall)
                        )
                conn.commit()

        return redirect(url_for('output'))

    return redirect(url_for('interview'))


@app.route('/interview/skip', methods=['POST'])
@login_required
def interview_skip():
    questions  = session.get('interview_questions', [])
    next_index = session.get('interview_question_index', 0) + 1
    session['interview_question_index'] = next_index

    if next_index >= len(questions):
        answers    = session.get('interview_answers', [])
        all_scores = [a['score'] for a in answers]
        overall    = round(sum(all_scores) / len(all_scores), 1) if all_scores else 0.0
        session['interview_score']    = overall
        session['interview_complete'] = True

        sess_id = session.get('interview_session_id')
        if sess_id:
            with get_db() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        'UPDATE interview_session SET overall_score=%s WHERE id=%s',
                        (overall, sess_id)
                    )
                conn.commit()
        return redirect(url_for('output'))

    return redirect(url_for('interview'))



@app.route('/output')
@login_required
def output():
    if not session.get('interview_complete'):
        return redirect(url_for('categories'))

    answers   = session.get('interview_answers', [])
    score     = session.get('interview_score', 0)
    category  = session.get('interview_category', 'hr')

    strengths  = [a for a in answers if a['score'] >= 70]
    weak_areas = [a for a in answers if a['score'] <  70]

    cat_labels = {
        'hr': 'HR Interview', 'python': 'Python Interview',
        'sql': 'SQL Interview', 'ml': 'Machine Learning', 'ds': 'Data Science',
    }

    if   score >= 80: 
        grade, grade_class = 'Excellent ',   'grade-good'
    elif score >= 60: 
        grade, grade_class = 'Good ',         'grade-mid'
    else:             
        grade, grade_class = 'Needs Practice ','grade-low'

    return render_template('output.html',
        **get_user_context(),
        answers=answers,
        score=score,
        grade=grade,
        grade_class=grade_class,
        category_label=cat_labels.get(category, 'Interview'),
        strengths=strengths,
        weak_areas=weak_areas,
        total_questions=len(answers),
    )


if __name__ == '__main__':
    app.run(debug=True)
