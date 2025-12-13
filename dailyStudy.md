# 12.13 学习内容
***君子性非异也，善假于物也***
## 汇编
```
   11.1  ZF 6
   11.2  PF 2
   11.3  SF 7
   
   11.4  CF 0
11.5  OF 11
11.6  adc指令
11.7  sbb指令
11.8  cmp指令
11.9  检测比较结果的条件转移指令
11.10 DF标志和串传送指令
```
### 积累
1.  PSW FLAG 寄存器
2.  ax,bx,cx,dx,si,di,bp,sp,ip,cs,ss,ds,es 13个寄存器
3.  ZF 結果為令0置为**1** , 直接影响标志寄存器的指令 **add,sub,mul,div,inc,or,and** ,不影响标志寄存器的指令 **mov,pop,push，inc，loop**
4.  PF 结果二进制一的个数为偶数的话置为**1**，奇数为 **0** 
5.  SF 结果为负SF为**1**，结果为正SF为**0**；负数为0，正数为1 补码表示有符号数 补码原码取反再加**1**
    无符号数没有意义 最高位是第七位，最低位是第零位
6.  CF 进位标志位 如果有进位置为**1**，没有进位置为**0** 对于**无符号数**运算有意义的标志位
7.  Debug 中的信息
```
    标志位       置为1       置为0
    OF           OV         NV
    SF           NG         PL
    ZF           ZR         NZ
    PF           PE         PO
    CF           CY         NC
    DF           DN         UP


```

8.  OF 溢出标志位 ***标志超出了机器所能表示的范围***将产生溢出，溢出置为**1**，没溢出置为**0** ，对于**有符号数**运算有意义的标志位
9.  **无符号数**运算看CF位即可，**有符号数**运算要看OF和SF
10. adc 指令 有符号加法指令 
    adc ax,4   ; (ax)+4+CF
    CF 值的含义 ，由adc指令前面的指令决定的，如果CF的值是被sub指令设置的那么含义就是借位，如果被add指令所设置的那就是进位
    CPU 设计adc指令的目的就是，进行加法的第二步运算
    add al,bl
    adc ah,bh     ;adc也会改变OF
    对任意大的数据进行加法运算
11. sbb指令，带借位除法指令 sbb ax,bx ;(ax) = (ax)-(bx)-CF
    对任意大的数进行减法运算
12. cmp指令，相当于sub但不会保存算出来的值，只会影响标志位的信息

无符号数
```

    ax  = bx  ZF = 1 && CF = 0
    ax != bx  ZF = 0
    ax  < bx  CF = 1 && ZF = 0                  ;产生借位
    ax >= bx  CF = 0
    ax  > bx  CF = 0 && ZF = 0
    ax <= bx  CF = 1 || ZF = 1

```
有符号数
```
    ax  = bx  ZF = 1 && SF = OF
    ax != bx  ZF = 0
    ax  < bx  ZF = 0 && CF != OF                ;产生借位
    ax >= bx  ZF = 1 || （ZF = 0 && SF = OF）
    ax  > bx  ZF = 0 && SF = OF
    ax <= bx  ZF = 1 || （ZF = 0 && SF != OF）
```
13. 条件转移指令

***检测标志位***  PSW寄存器
```
    je       等于则转移      ZF = 1                                 
    jne      不等于则转移    ZF = 0                                  
    jb       低于则转移      CF = 1
    jnb      不低于则转移    CF = 0
    ja       高于则转移      CF = 0 && ZF = 0
    jna      不高于则转移    CF = 1 || ZF = 1

```
***活学活用***

14. 串传送指令和 ***DF*** 标志位 
    movsb 的作用就是将 ds：si 指向内存单元中的字节送入es:di中，然后根据标志寄存器DF位的值
    将si和di递增或递减
    DF = 0    si = si + 1  正向传送
    Df = 1    si = si - 1  反向传送
      
    movsw  
    DF = 0    si = si + 2
    Df = 1    si = si - 2

    和rep指令配合使用
    rep movsb    ;根据CX 的值，重复执行
    
    cld 将DF 设置为0
    std 将DF 设置为1

    1. 传送的原始位置  ds:si;
    2. 传送的目的位置  es:di;
    3. 传送的长度       CX
    4. 传送的方向       DF

15. pushf 将标志寄存器的值压入栈
    popf  从栈中弹出数据，放入标志寄存器中
    



