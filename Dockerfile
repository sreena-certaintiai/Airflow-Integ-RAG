FROM astrocrpublic.azurecr.io/runtime:3.0-7
# This is an existing line
RUN /usr/local/bin/install-python-dependencies

# ---- This new line is correct ----
RUN python -m nltk.downloader punkt