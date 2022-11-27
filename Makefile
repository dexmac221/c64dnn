CFLAGS = -O -t $(TARGET) -I../cc65-floatlib/
LDFLAGS = -t $(TARGET)
CC = cc65
CA = ca65
LD = cl65

all: c64dnn.prg

c64dnn.prg: c64dnn.o 
	$(LD) $(LDFLAGS) -o c64dnn.prg main.o $(CC65_HOME)/lib/$(TARGET).lib ../cc65-floatlib/runtime.lib

c64dnn.o: main.c
	$(CC) $(CFLAGS) main.c ; $(CA) main.s

#c64dnn: main.c c64dnn.h ../cc65-floatlib/float.h ../cc65-floatlib/math.h
#	gcc -I../cc65-floatlib/ -O2 -W -Wall -Wextra -lm -o c64dnn main.c

# remove object files and executable when user executes "make clean"
clean:
	rm *.s *.o c64dnn.prg
