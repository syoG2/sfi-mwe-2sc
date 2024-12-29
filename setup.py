from distutils.core import setup

setup(
    name="sfimwe2sc",
    version="0.1.0",
    description="sfimwe2sc",
    author="Kosuke Yamada",
    packages=["sfimwe2sc", "lib"],
    package_dir={
        "sfimwe2sc": "sfimwe2sc",
        "lib": "lib",
    },
)

# setup(
#     name="lib",
#     version="0.0.0",
#     description="My Library",
#     author="Kosuke Yamada",
#     packages=["lib"],
#     package_dir={
#         "lib": "lib",
#     },
# )
