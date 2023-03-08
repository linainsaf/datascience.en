## Introduction 

Welcome to the Database Management Systems course! In this course, you will learn how to use SQL to work with databases (extract, insert, delete and analyse data). Databases are an essential part of modern computing, and they are used in everything from social media applications to e-commerce websites to scientific research.

## Course goals 

This course will cover database basics, installation of MYSQL, various database types, database designs, SQL statements, and more. Here are some main goals as a developer for mastering databases:

* Understanding data modeling: The ability to design and implement effective data models is crucial for building scalable and efficient databases. This includes knowledge of different types of databases, data structures, and normalization techniques.
* Proficiency in SQL: SQL (Structured Query Language) is the standard language used for querying, manipulating, and managing data in databases. As a developer, it is important to have a strong understanding of SQL and its various functions.
* Database administration skills: Database administration involves tasks such as installing, configuring, and maintaining databases. A developer who is proficient in database administration can optimize performance, troubleshoot issues, and ensure the security of their databases.
* Knowledge of database architecture: Understanding the architecture of databases and how they interact with other systems is essential for building and integrating applications with databases.

## Table of content 

1. Introduction
    * Overview of the main topics that will be covered in the course and goals
    * Explanation of what a database is and why they are important
2. Introduction to relational DBMS
    * What is a RDBMS
    * What is MySQL
    * MySQL installation
3. Database designs basics
    * Explanation of tables and keys in databases
    * Explanation of SQL language and its uses
    * Overview of different SQL Statements
    * Explanation of DDL language and its uses
    * Overview of different DDL Statements
4. Using MySQL Workbench
    * Data servers
    * MySQL/RDBMS Concepts
    * Overview of different TCL statements
    * Overview of different DML statements
5. MySQL users
    * Introduction
    * Users selection
    * User privileges
6. Managing my database
    * Creating database
    * Creating tables
    * Introduction to MySQL Data Types
    * MySQL Clauses
    * Retreive data from single tables
7. Other statements
    * Overview of the Groupby statement
    * Joins in SQL
    * Creating Views with Aggregate functions
    * Constraints
    * Indexes
    * Select Statement
8. Advanced topics
    * Subqueries
    * Introduction to Extraction Transformation and Loading
    * ETL Process
    * Introduction to NoSQL databases

Let's get started 🥳 

## Why database ? 

Imagine you work for a large online retailer, and your company needs to manage thousands of products, customers, and orders every day. You could store this data in an Excel spreadsheet, but as your company grows, this becomes increasingly difficult and time-consuming. You might have dozens or even hundreds of spreadsheets, each containing different pieces of data, making it difficult to find the information you need quickly. In addition, spreadsheets are not ideal for handling large amounts of data, and they can become slow and unwieldy as the size of the dataset grows.

<br />

This is where databases come in. A database is an organized collection of data that is designed to be easy to access, manage, and update. With a database, you can store all your company's data in one place, making it easy to find the information you need quickly. Databases are also designed to handle large amounts of data efficiently, so you can work with massive datasets without running into performance issues.

## Types of Databases

There are many different types of databases, each with its own strengths and weaknesses. Some of the most common types of databases include:

* Relational databases: These databases organize data into tables with columns and rows, similar to a spreadsheet. They are the most common type of database and are used in many different applications.
* NoSQL databases: These databases are designed to handle large amounts of unstructured or semi-structured data. They are commonly used in big data applications, such as social media analytics and scientific research.
* Object-oriented databases: These databases store data as objects, which can be manipulated using object-oriented programming techniques. They are commonly used in software development and data modeling.

Some examples of day-to-day use cases for each type of database : 

### Relational databases

* Customer database example : A business might use a relational database to keep track of customer information, such as their name, contact details, and purchase history. They could then use this information to target customers with personalized marketing campaigns based on their previous purchases.
* Inventory management example : A store might use a relational database to manage their inventory, with one table for products and another table for suppliers. They could then use SQL queries to quickly retrieve information on which products are in stock, which products are selling quickly, and which suppliers they need to contact to restock their inventory.
* Employee scheduling example : A company might use a relational database to manage employee schedules, with one table for employees and another table for shifts. They could then use SQL queries to quickly retrieve information on which employees are available to work on a particular day or time, and which shifts still need to be filled.

### NoSQL databases 

* Social media analytics: Social media platforms like Facebook and Twitter use NoSQL databases to store and analyze massive amounts of user data, such as likes, comments, and shares. This allows them to quickly retrieve and analyze user data to provide better ad targeting and personalized content.
* Internet of Things (IoT) devices: IoT devices like smart thermostats and security cameras generate a huge amount of data, which can be stored and analyzed in NoSQL databases. This allows manufacturers to track device usage patterns, identify and fix bugs, and improve device performance over time.
* Gaming: Many video games use NoSQL databases to store player data, such as character stats and in-game achievements. This allows players to continue their game progress across different devices, and enables game developers to quickly retrieve and analyze player data to identify areas for improvement.

### Object-oriented databases

* Geolocation data: Companies that rely on geolocation data, such as mapping and navigation services, often use object-oriented databases to store and retrieve this data. This allows them to quickly retrieve and analyze large amounts of geolocation data in real-time.
* E-commerce: An e-commerce website might use an object-oriented database to store and manage product information, such as product images and descriptions. This allows them to easily update and manage product information across multiple platforms, such as their website, mobile app, and social media.
* Medical records: Hospitals and healthcare providers often use object-oriented databases to manage patient medical records, which can include a wide range of data types, such as images, test results, and diagnoses. This allows healthcare providers to easily access and update patient information, and can help improve patient care and outcomes.

In this course, we will be focusing on relational databases, specifically MySQL, which is a popular open-source database management system. We will also introduce the concept of NoSQL databases.



## What are DBMS ?

A Database Management System (DBMS) is defined as the software system that allows users to define, create, maintain and control access to the database. DBMS makes it possible for end users to create, read, update and delete data in database. A DBMS serves as an interface between an end-user and the database. DBMS Examples: MySQL, Microsoft Access, SQL Server, FileMaker, Oracle.

<br />

Advantages of DBMS : 

- Improved Data Availability
- Improved Data Security Improved
- data integration Improved decision making
- Increased end-user productivity Simplicity




