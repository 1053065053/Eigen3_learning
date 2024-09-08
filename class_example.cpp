#include <iostream>
using namespace std;

class Animal {
public:
    Animal() {
        cout << "Animal created" << endl;
    }
    
    virtual void speak() {
        cout << "Animal sound" << endl;
    }

    virtual ~Animal() {
        cout << "Animal destroyed" << endl;
    }
};

class Dog : public Animal {
public:
    Dog() {
        cout << "Dog created" << endl;
    }

    void speak() override {
        cout << "Woof!" << endl;
    }

    ~Dog() {
        cout << "Dog destroyed" << endl;
    }
};

int main() {
    Animal* myDog = new Dog(); // 创建一个 Dog 对象
    myDog->speak();            // 调用重写的函数
    delete myDog;             // 删除对象，调用析构函数
    return 0;
}