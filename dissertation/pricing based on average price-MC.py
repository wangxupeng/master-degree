import scipy as sp
s0=50
k=40
T=0.5
r=0.1
sigma=0.3
n_simulation=100000
n_steps=40
dt=T/n_steps
call=sp.zeros([n_simulation], dtype=float)
for j in range(0, n_simulation):
    sT=s0
    total=0
    for i in range(0,int(n_steps)):
        e=sp.random.normal()
        sT*=sp.exp((r-0.5*sigma*sigma)*dt+sigma*e*sp.sqrt(dt))
        total+=sT
    price_average=total/n_steps
    call[j]=max(price_average-k,0)
call_price=sp.mean(call)*sp.exp(-r*T)
print('call price = ', round(call_price,3))
