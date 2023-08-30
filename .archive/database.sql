-- create a new role
CREATE ROLE read_only_all;
GRANT SELECT ON ALL TABLES IN SCHEMA public 
TO read_only_all;

GRANT INSERT, UPDATE, SELECT, DELETE
ON ALL TABLES IN SCHEMA public 
TO read_write_all;

-- create a new user
CREATE USER oytun_demirbilek WITH login password 'change_me_2413' nosuperuser inherit nocreatedb nocreaterole noreplication valid until 'infinity' IN GROUP read_only_all;
ALTER USER oytun_demirbilek WITH password 'your_new_password_with_quotes';