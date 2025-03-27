# dashboard.py
from signac_dashboard import Dashboard
from signac_dashboard.modules import *

modules = [
    StatepointList(),
    DocumentList(context="JobContext"),
    DocumentList(context="ProjectContext"),
    ImageViewer(context="JobContext"),
    ImageViewer(context="ProjectContext"),
    Schema(),
    FileList(context="JobContext"),
]

class MyDashboard(Dashboard):
    def job_title(self, job):
        if int(job.sp["num_flattener"]) > 0:
            title_str = "Flat"
        else:
            title_str = "Rounded"
        
        return "{}, Aspect Ratio = {}, v0 = {}, Freedom Ratio = {}".format(title_str, job.sp["aspect_rat"], job.sp["v0"], job.sp["freedom_rat"])
        

if __name__ == "__main__":
   #Dashboard(modules=modules).main()
   MyDashboard(modules=modules).main()
