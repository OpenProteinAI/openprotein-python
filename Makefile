VERSION ?= 0.5.2
SHELL := /bin/bash

.PHONY: releasehere
releasegit:
	# Update Pixi version
	pixi project version set $(VERSION)

	# Update version in meta.yaml
	# sed -i'' 's/^  version: .*/^  version: "$(VERSION)"/' anaconda_build/meta.yaml
	perl -i -pe's/^  version: .*/  version: "$(VERSION)"/' anaconda_build/meta.yaml

	# Commit changes
	git add pyproject.toml anaconda_build/meta.yaml
	git commit -m "Bump version to $(VERSION)"

	# Create a new tag
	git tag -a v$(VERSION) -m "Release version $(VERSION)"

	# Push changes and tag
	git push origin HEAD
	git push origin v$(VERSION)


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

condabld:
	micromamba activate openprotein-sdk-build
	conda build -c conda-forge --numpy 2.1 ./anaconda_build

proddocs:
	cd apidocs && make clean && make html 
	aws s3 sync apidocs/build/html s3://openprotein-docs-prod/api-python/
	aws cloudfront create-invalidation --distribution-id E1CUT1CP31D5NK --paths "/*" 

devdocs: 
	cd apidocs && make clean && make html 
	aws s3 sync apidocs/build/html s3://openprotein-docs-dev/api-python/
	aws cloudfront create-invalidation --distribution-id E3SMW2DYY71HHW --paths "/*"
