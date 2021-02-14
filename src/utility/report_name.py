import subprocess

def get_report_name(basename):
    sub = subprocess.Popen("echo \"$(git rev-parse HEAD)-$(date '+%Y.%m.%d-%H.%M.%S')\"",
                                  shell=True,
                                  stdout=subprocess.PIPE)
    gitRevId_timestamp = sub.stdout.read().decode("utf-8").strip()
    return "../results/" + basename + "_git_" + gitRevId_timestamp