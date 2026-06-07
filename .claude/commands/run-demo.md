# /run-demo

Launch the Gradio health companion app in development mode and verify all four tabs load.

Steps:
1. Check that `app/main.py` exists
2. Run `python app/main.py` and confirm the local URL appears in output
3. Use the browser tool to open the URL and verify the four tabs: Check-in, Camera/Read, Doctor Brief, History
4. Report any import errors or missing model files clearly

If models are not yet downloaded, note which ones are missing and skip model-loading; test only the UI skeleton.
