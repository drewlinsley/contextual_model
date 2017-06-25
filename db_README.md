#0. Prepare postgresql databse
	a. sudo apt-get install postgresql libpq-dev postgresql-client postgresql-client-common #Install posetgresql with online installer
	b. sudo -i -u postgres #goes into postgres default user
	c. psql postgres #enter the postgres interface
	d. create role context WITH LOGIN superuser password 'serrelab'; #create the admin
	e. alter role context superuser; #ensure we are sudo
	f. create database context with owner context; #create the database
	g. \q #quit
#1. Insert table for hp optimization
	a. psql context -h 127.0.0.1 -d context #log into the database with the admin credentials
	b. create table hpcombos (_id bigserial primary key, lesions varchar, alpha float, beta float, mu float, nu float, gamma float, delta float, f3a float, f3b float, f4 float, f5 float, f7 float, tbp float, tbtcso float, bw float); 
	c. create table status (_id bigserial primary key, fig_names varchar, working boolean default false); 
#Or just run: python start_hp_optims.py

#Pachaya instructions:
	a. You need to clone https://github.com/serre-lab/hmax/ and use the branch vbw.
	b. A lot of the scripts unfortunately at the moment add that path with sys and a hardcoded path name. You can either make this more general and recode things or just change the path to reflect where your hmax directory is.
