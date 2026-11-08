# Deploy FourierForensics on Streamlit Community Cloud

The application runs directly from this repository. It needs no API keys,
model downloads, or external datasets. Deployment requires signing in to your
own Streamlit Community Cloud account and granting access to this repository.

## Create the app

1. Open [Streamlit Community Cloud](https://share.streamlit.io/new).
2. Sign in with GitHub and choose **Create app** (or **New app**).
3. Choose deployment from an existing GitHub repository and enter:

   | Setting | Value |
   | --- | --- |
   | Repository | `ghsihem228-ship-it/FourierForensics` |
   | Branch | `main` |
   | Main file path | `app.py` |
   | Python version, under Advanced settings | `3.11` |

4. Choose an available app subdomain. No secrets are required.
5. Click **Deploy** and wait for dependency installation and app startup.

Community Cloud installs `requirements.txt` from the repository root. OpenCV
uses the headless package, so no desktop graphics packages are needed.
The server upload limit is configured to 25 MiB in `.streamlit/config.toml`.
The loader also enforces a 16-megapixel limit. Full-resolution analysis can be
slow on shared hosting; the included 256-pixel examples are a useful first check.

## Verify the public deployment

- Open the assigned public URL and confirm that the generated splice example
  displays the original, spectrum, noise map, and detector explanations.
- Select the single-JPEG example and open the diagnostics tab.
- Upload a valid JPEG and download the analysis JSON.
- Open the URL on a phone to check the responsive layout.

After verification, put the assigned URL in GitHub's **About → Website** field
and link to it from the README. Use the URL actually assigned by Streamlit;
the project does not reserve a subdomain automatically.

Subsequent pushes to `main` are picked up by Community Cloud. GitHub Actions
runs the numerical, detector, integration, UI, and evaluation tests on Python
3.11 and 3.12 independently of the hosted app.

## If startup fails

Use **Manage app → Logs** to find the failing dependency or Python traceback.
Confirm that the main file is `app.py`, Python is 3.11, and installation uses
the root `requirements.txt`. That file contains runtime dependencies;
`requirements-dev.txt` and `requirements-lock.txt` also contain development
tools and are unnecessary for this deployment.
