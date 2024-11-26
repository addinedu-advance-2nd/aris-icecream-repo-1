# %%
import sqlite3
#create table - Order
con_Order = sqlite3.connect('/home/hjpark/dev_hjp/aris-repo-1/DB/DB_list/Orders.db')
cur_Order = con_Order.cursor()
for row in cur_Order.execute('''SELECT * FROM Orders'''):
        print(row)

con_Order.close()

# %%
import sqlite3
con_Membership = sqlite3.connect('./DB_list/Membership.db')
cur_Membership = con_Membership.cursor()
for row in cur_Membership.execute('''SELECT * FROM Membership'''):
        print(row)

con_Membership.close()

# %%
import sqlite3
con_Reward_preference = sqlite3.connect('./DB_list/Reward_preference.db')
cur_Reward_preference = con_Reward_preference.cursor()
for row in cur_Reward_preference.execute('''SELECT * FROM Reward_preference'''):
        print(row)

con_Reward_preference.close()

# %%
import sqlite3
con_Robot_incident = sqlite3.connect('./DB_list/Robot_incident.db')
cur_Robot_incident = con_Robot_incident.cursor()
for row in cur_Robot_incident.execute('''SELECT * FROM Robot_incident'''):
        print(row)

con_Robot_incident.close()

# %%
