#include <thread>
#include <iostream>

void say_hello(int id) {
	std::string msg = "hello from thread" + std::to_string(id) + "\n";
	std::cout << msg;
}


int main() {
	std::thread t1(say_hello, 1);
	std::thread t2(say_hello, 2);

	t1.join();
	t2.join();

	std::cout << "Threads done running" << std::endl;

	return 0;
}
