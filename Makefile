VERSION ?= 0.2.3

.PHONY: release
release:
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
