VERSION ?= 0.4.0
SHELL := /bin/bash

.PHONY: releasehere
releasegit:
	# Update Poetry version
	poetry version $(VERSION)

	# Update version in meta.yaml
	sed -i 's/^  version: .*/  version: "$(VERSION)"/' anaconda_build/meta.yaml

	# Commit changes
	git add pyproject.toml anaconda_build/meta.yaml
	git commit -m "Bump version to $(VERSION)"

	# Create a new tag
	git tag -a v$(VERSION) -m "Release version $(VERSION)"

	# Push changes and tag
	git push origin HEAD
	git push origin v$(VERSION)

	curl -X PURGE https://camo.githubusercontent.com/125a275204c801f733fd69689c1e72bde9960a1e193c9c46299d848373a52a93/68747470733a2f2f616e61636f6e64612e6f72672f6f70656e70726f7465696e2f6f70656e70726f7465696e2d707974686f6e2f6261646765732f76657273696f6e2e737667


releasehere:
	# Update Poetry version
	poetry version $(VERSION)

	# Update version in meta.yaml
	sed -i 's/^  version: .*/  version: "$(VERSION)"/' anaconda_build/meta.yaml

	# Commit changes
	git add pyproject.toml anaconda_build/meta.yaml
	git commit -m "Bump version to $(VERSION)"

	# Create a new tag
	git tag -a v$(VERSION) -m "Release version $(VERSION)"

	# Push changes and tag
	git push origin HEAD
	git push origin v$(VERSION)

	# pypi 
	poetry build 
	poetry publish 

	#conda 
	source activate bld && conda build ./anaconda_build
	
	curl -X PURGE https://camo.githubusercontent.com/125a275204c801f733fd69689c1e72bde9960a1e193c9c46299d848373a52a93/68747470733a2f2f616e61636f6e64612e6f72672f6f70656e70726f7465696e2f6f70656e70726f7465696e2d707974686f6e2f6261646765732f76657273696f6e2e737667

proddocs:
	cd apidocs && make clean && make html 
	aws s3 sync apidocs/build/html s3://openprotein-docs-prod/api-python/
	aws cloudfront create-invalidation --distribution-id E1CUT1CP31D5NK --paths "/*" 

devdocs: 
	cd apidocs && make clean && make html 
	aws s3 sync apidocs/build/html s3://openprotein-docs-dev/api-python/
	aws cloudfront create-invalidation --distribution-id E3SMW2DYY71HHW --paths "/*"
