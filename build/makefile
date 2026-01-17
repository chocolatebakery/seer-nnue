#  * Stormphrax, a UCI chess engine
#  * Copyright (C) 2024 Ciekce
#  *
#  * Stormphrax is free software: you can redistribute it and/or modify
#  * it under the terms of the GNU General Public License as published by
#  * the Free Software Foundation, either version 3 of the License, or
#  * (at your option) any later version.
#  *
#  * Stormphrax is distributed in the hope that it will be useful,
#  * but WITHOUT ANY WARRANTY; without even the implied warranty of
#  * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#  * GNU General Public License for more details.
#  *
#  * You should have received a copy of the GNU General Public License
#  * along with Stormphrax. If not, see <https://www.gnu.org/licenses/>.


EXE = seer
CXX = g++

CXXSTANDARD = 20
EVALFILE = atomic.nnue
OPSLIMIT = 1000000000

CXXSRC += $(wildcard ../src/*.cc )
CXXSRC += $(wildcard ../src/search/*.cc)
CXXSRC += $(wildcard ../src/chess/*.cc)
CXXSRC += $(wildcard ../src/engine/*.cc)
CXXSRC += $(wildcard ../src/eval/*.cc)
CXXSRC += ../syzygy/tbprobe.cc

INCLUDE = ../include
INCBIN = ../incbin
SYZYGY = ../syzygy

BASECXXFLAGS += -std=c++$(CXXSTANDARD)
BASECXXFLAGS += -O3 -g -DNDEBUG -march=native -mtune=native -fopenmp
BASECXXFLAGS += -Wall -Wextra -pedantic
BASECXXFLAGS += -fconstexpr-ops-limit=$(OPSLIMIT)
BASECXXFLAGS += -DEVALFILE=\"$(EVALFILE)\"
CXXFLAGS += $(BASECXXFLAGS)


CPPFLAGS += -MMD
CPPFLAGS += -I$(INCLUDE)
CPPFLAGS += -I$(INCBIN)
CPPFLAGS += -I$(SYZYGY)

LDFLAGS = -lpthread

CXXOBJECTS += $(CXXSRC:%.cc=%.o)
CXXDEPENDS += $(CXXSRC:%.cc=%.d)

.PHONY: default
default: flto

.PHONY: pgo
pgo:
	$(MAKE) clean
	$(MAKE) profileclean
	$(MAKE) profilegenerate EXE=$(EXE) CXX=$(CXX) EVALFILE=$(EVALFILE)
	./$(EXE) bench
	$(MAKE) clean
	$(MAKE) profileuse EXE=$(EXE) CXX=$(CXX) EVALFILE=$(EVALFILE)
	$(MAKE) profileclean

.PHONY: profilegenerate
profilegenerate: CXXFLAGS += -fprofile-generate
profilegenerate: flto

.PHONY: profileuse
profileuse: CXXFLAGS += -fprofile-use
profileuse: flto

.PHONY: flto
flto: CXXFLAGS += -flto -flto-partition=one -fwhole-program
flto: LDFLAGS += -flto=jobserver
flto: binary

.PHONY: plain
flto: binary

.PHONY: binary
binary: $(CXXOBJECTS)
	+$(CXX) $(CXXFLAGS) -o $(EXE) $^ $(LDFLAGS)

.PHONY: clean
clean:
	rm -f $(CXXOBJECTS) $(CXXDEPENDS)

.PHONY: profileclean
profileclean:
	rm -f $(CXXSRC:%.cc=%.gcda)


-include $(CXXDEPENDS)
