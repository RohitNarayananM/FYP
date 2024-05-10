import sqlite3
import matplotlib.pyplot as plt
import os

if not os.path.exists("data.txt"):
    conn = sqlite3.connect("CVEfixes.db")
    c = conn.cursor()
    l1 = c.execute("SELECT LENGTH(code) FROM method_change").fetchall()
    l2 = c.execute("SELECT LENGTH(code_before) FROM file_change").fetchall()

    with open("data.txt", "w") as f:
        f.write(str(l1) + "\n" + str(l2))
else:
    with open("data.txt", "r") as f:
        l1, l2 = f.read().split("\n")
        l1 = eval(l1)
        l2 = eval(l2)

l1.sort()
l2.sort()
l1 = l1[:50000]
l2 = l2[:30000]

# Plot both lists
# plt.plot(range(len(l1)), l1, label="Method Change")
plt.plot(range(len(l2)), l2, label="File Change")
plt.legend()
plt.show()
