import re

demo = "jdlgjl435453dsgfd dhogd dsghk asf53 ds44fg 43 3132"
res = re.split(r'[0-9]*', demo)
print(res)
