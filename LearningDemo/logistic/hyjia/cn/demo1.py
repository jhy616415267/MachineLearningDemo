# _*_coding:UTF-8 _*_


def Grandient_Ascent_test():
    def f_prime(x_old):
        return -2*x_old+4
    x_old = -1
    x_new = 0
    alpha = 0.01
    precision = 0.000000001
    while abs(x_new-x_old)>precision:
        x_old = x_new
        x_new = x_old+alpha*f_prime(x_old)
    print(x_new)
    
if __name__ == '__main__':
    Grandient_Ascent_test()
