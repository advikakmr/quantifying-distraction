import roboflow

rf = roboflow.Roboflow(api_key="294DizQOGWKUWFsIjLcn")
model = rf.workspace().project("my-first-project-mpgwh").version("1").model
model.download() # Downloads 'weights.pt' to your local folder