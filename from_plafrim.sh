# scp {plafrim_launcher,main,check_finished}.py hchateau@plafrim:/beegfs/hchateau/opportunistic-pfc
# scp -r opportunistic_pfc hchateau@plafrim:/beegfs/hchateau/opportunistic-pfc/
rsync --exclude='__pycache__' -ar hchateau@plafrim:/projets/episodic-rl/EC/logs/$1 ./logs
