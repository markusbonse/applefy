
pypi:
	python setup.py sdist bdist_wheel
	twine check dist/*
	twine upload dist/*

docs-clean:
	rm -rf docs/build
	cd docs
	$(MAKE) -C docs html
