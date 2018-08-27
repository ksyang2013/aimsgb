#!/usr/bin/env python
#Ab-Initio Materials Simulation (AIMS) Software Package Developed in Prof. Yang's Research Group

pyc: 
	rm -rf build/
	cd main && python -m py_compile *.py
	cd xio && python -m py_compile *.py
	cd xlibs && python -m py_compile *.py
	cd xinstructions && python -m py_compile *.py
	mkdir build
	cp -rf main/*pyc build
	cp -rf xio/xio.pyc build
	cp -rf xlibs/xlibs.pyc build
	cp -rf xinstructions/xinstructions.pyc build
	cd build && python aims.pyc -h

nano:
	ln -s /home/kesong/AIMS/aims/main/aims.py ~/bin/aims

tscc:
	ln -s /projects/kyang-group/AIMS/aims/main/aims.py /projects/kyang-group/bin/aims

binary:
	cd build && (python ../freeze/freeze.py ../main/aims.py) && make
	cp build/aims ./bin/

wget:
	wget http://materials.ucsd.edu/code/aims.tgz.gpg
	gpg aims.tgz.gpg

clean:
	rm -f *~
	rm -f *.pyc
	rm -f *.pyg
	rm -f */*~
	rm -f */*.pyc
	rm -f */*.pyg
	rm -f */*.py.bak
	rm -f */*/*~
	rm -f */*/*.pyc
	rm -f */*/*.pyg
	rm -f */*/*.py.bak
	rm -f */*/*/*~
	rm -f */*/*/*.pyc
	rm -f */*/*/*.pyg
	rm -f */*/*/*.py.bak
	#rm -r build/*
	if [ -f ./bin/aims ]; then rm -r ./bin/aims; fi
