str1="~!@#$%^&*()_=-/,.?<>;:[]{}\|"
has_num=0
has_str=0
has_alp=0
t="y"
while t == "y":
        pw = input("请输入密码:")
        length = len(pw)
        while ( pw.isspace() or length == 0 ) :
                pw = input("您输入的密码为空,请重新输入:")
        for i in pw:
                if i in str1:
                        has_str=1
                if i.isdigit():
                        has_num=1
                if i.isalpha():
                        has_alp=1
        has = has_num + has_alp + has_str
        if length <= 8 and has == 1:
                level = "低"
        if length > 8 and has == 2 :
                level = "中"
        if length > 8 and has == 3 and pw[0].isalpha():
                level = "高"
        print("您的密码等级为"+ level)
        if level == "高":
                print("你猴塞雷!")
        else:
                print("""请按以下方式提升您的密码安全级别：
        1.密码必须由数字、字母及特殊字符三种组合
        2.密码只能由字母开头
        3.密码长度不能低于16位""")
        t = input("如果想再创建一个密码就输入y,不想玩了输入任意值退出:")
