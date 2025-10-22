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
        if int(job.sp["active_angle"]) == 0:
            title_str = "End Coated"
        elif int(job.sp["active_angle"]) == 90:
            title_str = "Side Coated"
        else:
            title_str = "Angle Coated"
        
        return "{}, Confinement = {}, torque mag = {}, wall_R = {}".format(title_str, job.sp["confinement"], job.sp["torque_mag"], job.sp["wall_R"])
        

if __name__ == "__main__":
   #Dashboard(modules=modules).main()
   MyDashboard(modules=modules).main()
