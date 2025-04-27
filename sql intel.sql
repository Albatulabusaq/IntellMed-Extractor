CREATE DATABASE my_sql;
USE my_sql;

CREATE TABLE User (
  User_id INT AUTO_INCREMENT PRIMARY KEY,
  User_name VARCHAR(255),
  Phone VARCHAR(50),
  Password VARCHAR(255),
  first_name VARCHAR(100),
  last_name VARCHAR(100)
); 

CREATE TABLE Admin (
  AdminID INT AUTO_INCREMENT PRIMARY KEY,
  Admin_name VARCHAR(255),
  Password VARCHAR(255),
  Technical_Support VARCHAR(255)
);
CREATE TABLE Document (
  Document_ID INT AUTO_INCREMENT PRIMARY KEY,
  Document_Name VARCHAR(255),
  Format VARCHAR(50),
  Text TEXT,
  PDF VARCHAR(255),
  DOCX VARCHAR(255)
);
CREATE TABLE Extracted_Data (
  Entity_Id INT,
  Data_Id INT,
  Entity_Type VARCHAR(255),
  Outcoms VARCHAR(255),
  Intervention VARCHAR(255),
  Participation VARCHAR(255),
  Comparators VARCHAR(255),
  Document_Id INT,
  User_Id INT,
  Summary TEXT,
  PRIMARY KEY (Entity_Id, Data_Id),
  FOREIGN KEY (Document_Id) REFERENCES Document(Document_ID),
  FOREIGN KEY (User_Id) REFERENCES User(User_id)
);

CREATE TABLE Export_Format (
  Export_ID INT AUTO_INCREMENT PRIMARY KEY,
  CSV VARCHAR(255),
  PDF VARCHAR(255)
);
SHOW TABLES;
DESCRIBE User;

ALTER TABLE User
MODIFY User_name VARCHAR(255) NOT NULL,
MODIFY Phone VARCHAR(50) NOT NULL,
MODIFY Password VARCHAR(255) NOT NULL;

DESCRIBE admin;
DESCRIBE document;
DESCRIBE user;
DESCRIBE export_format;
DESCRIBE extracted_data;


ALTER TABLE user CHANGE phone email VARCHAR(255);

ALTER TABLE User ADD CONSTRAINT unique_email UNIQUE (email);
ALTER TABLE User ADD CONSTRAINT unique_Password UNIQUE (Password);

ALTER TABLE extracted_data
  ADD CONSTRAINT fk_document_id FOREIGN KEY (Document_Id) REFERENCES document(Document_Id),
  ADD CONSTRAINT fk_user_id FOREIGN KEY (User_Id) REFERENCES user(User_Id);
  





