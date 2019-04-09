# Water monitoring system

Tested on Raspberry Pi 3 Model B Plus Rev 1.3

# Table of contents

* [Requirements](#requirements) 
* [Python environment](#python-environment)
* [Database](#database)
* [Peak detection](#peak-detection)
* [Launch](#launch)

## Requirements

  - Raspbian
  - sudo apt install python3-venv libpq-dev python3-dev libjpeg-dev
 
It is highly recommended to calibrate the compass which is extremely sensitive.
https://www.raspberrypi.org/documentation/hardware/sense-hat/

## Python environment

Virtualenv is used to handle dependencies that are stored in the requirements.txt file.

```ssh
python3 -m venv myvenv
source venv/bin/activate
pip install -r requirements.txt
```

## Database

The data from the water consumption is stored in a database, Postgresql is used here.

Connect to postgres as postgres default user.
```ssh
sudo -u postgres psql postgres
```
Create the database and the pi user affiliated to it.
```sql
create database watermonitor;
create user pi with encrypted password 'piwater';
grant all privileges on database watermonitor to pi;
```
Connect as pi to watermonitor database.
```ssh
sudo -u pi psql -d watermonitor
```
Create water_consumption table. Index is used on timestamp to have quicker response on operations filtering time.
```sql
CREATE TABLE water_consumption (timestamp bigint, value float);
CREATE INDEX index_timestamp ON water_consumption (timestamp);
```

## Peak detection

The peak detection code allows to detect peaks from a signal sample, ignoring noises thanks to a set delimiter.

> Inspired by https://gist.github.com/endolith/250860

## Launch

To start the program, you must beforehand activate the python environment, then launch the pimonitor

```ssh
source venv/bin/activate
python3 pimonitor.py
```

The use of the screen application is recommended. When connecting in ssh to the raspberry and starting the application, exiting the ssh window will lose the focus on the program running. Thanks to the screen application, you can return to the state of the running application.

```ssh
screen python pimonitor.py
```

To detached the focus: ctrl+a then ctrl+d
To reattach the focus: screen -r


License
----

MIT
