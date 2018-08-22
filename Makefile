# Makefile
env: env/bin/activate

env/bin/activate: requirements.txt
	virtualenv -p python3 env
	. env/bin/activate; pip install -r requirements.txt
	touch env/bin/activate

clean:
	rm -rf env
