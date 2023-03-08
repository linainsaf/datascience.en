## What are RDBMS ?

A relational database organizes data into tables which can be linked—or related—based on data common to each. This capability enables you to retrieve an entirely new table from data in one or more tables with a single query. Data is represented in the terms of rows/records and columns. In a relational database, each row in the table is a record with a unique ID called the key.

## What is MySQL ?

MySQL is one of the most recognizable technologies in the modern big data ecosystem. Often called the most popular database and currently enjoying widespread, effective use regardless of industry, it’s clear that anyone involved with enterprise data or general IT should at least aim for a basic familiarity of MySQL.

<br />

MySQL is a relational database management system (RDBMS) developed by Oracle that is based on structured query language (SQL).

<br />

MySQL is integral to many of the most popular software stacks for building and maintaining everything from customer-facing web applications to powerful, data-driven B2B services. Its open-source nature, stability, and rich feature set, paired with ongoing development and support from Oracle, have meant that internet-critical organizations such as Facebook, Flickr, Twitter, Wikipedia, and YouTube all employ MySQL backends.

### MySQL client and CLI

Think of a database like a big organized warehouse full of information. In order to get information from the warehouse or add information to it, we need a way to talk to it. 

<br />

That's where the MySQL client comes in. It's like the messenger that we use to talk to the warehouse, and it allows us to read or update information stored in the database.

<br />

We use a command line interface (CLI) to interact with the MySQL client, which allows us to send specific commands to the database. A command-line interface (CLI) is a text-based interface used to interact with a computer's operating system or software by typing commands in a terminal window. It allows you to perform various tasks such as managing files and directories, executing programs, and interacting with databases, by entering commands and receiving text-based output.

<br />

In the context of MySQL, the CLI is a tool that enables you to interact with a MySQL database from a terminal window or a graphic interface by typing SQL commands. This means you can create, modify, and query your databases without using a graphical user interface. The CLI is a powerful tool that gives you fine-grained control over your database, but it requires some knowledge of SQL commands and syntax to use effectively.

### Vocabulary 

Here's an additional point on the differences between MySQL client, server, and SQL:

* MySQL client is a command-line tool that allows you to interact with the MySQL server, execute SQL queries, and manage databases.
* MySQL server is the software that stores and manages databases, and allows multiple clients to connect to it and perform operations on the databases.
* SQL (Structured Query Language) is the language used to interact with relational databases like MySQL, and it provides a standardized syntax for creating, modifying, and querying databases.

### Advantages of MySQL

* MySQL is secure as it consists of a solid data security layer to protect sensitive data from intruders and passwords in MySQL are encrypted
* MySQL is free and open source
* MySQL is compatible with most platforms including Windows, MacOS and Linux
* MySQL provides the ability to run the clients and the server on the same computer or on different computers, via internet or local network.
* MySQL has a unique storage engine architecture which makes it faster, cheaper and more reliable.
* MySQL gives developers higher productivity by using views, Triggers and Stored procedures
* MySQL is simple and easy to use. You can build and interact with MySQL with only the basic knowledge of MySQL and a few simple SQL statements.
* MySQL is scalable and capable of handling more than 50 million rows. This is enough to handle almost any amount of data. Although the default file size limit is 4GB but it can
be increased to 8TB.

### Disadvantages of MYSQL

* MySQL is not very efficient in handling very large databases.
* MySQL doesn’t have as good a developing and debugging tool as compared to paid databases.
* MySQL versions less than 5.0 do not support COMMIT, stored procedure and ROLE.
* MySQL is prone to data corruption as it inefficient in handling transactions.
* MySQL does not support SQL check constraints.

## Installing MySQL

To install MySQL on your computer, please visit their [**website**](https://dev.mysql.com/downloads/mysql/) And download the free and open source community edition of MYSQL. 

* For Windows select “Installer for Windows”
* For Mac select “MacOS”
* For Linux select “APT” or “YUM”

Once downloaded, run the installer and follow the instructions to install MYSQL on your computer then launch the MYSQL Workbench


