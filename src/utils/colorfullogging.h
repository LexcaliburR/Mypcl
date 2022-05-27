/*
 * @Author: Lexcaliburr lishiqi0111@gmail.com
 * @Date: 2022-05-27 22:34:26
 * @LastEditors: Lexcaliburr lishiqi0111@gmail.com
 * @LastEditTime: 2022-05-27 22:34:33
 * @FilePath: /Mypcl/src/utils/colorfullogging.h
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置
 * 进行设置:
 */
#pragma once

#include <glog/logging.h>
#include <chrono>
#include <string>
#include <iostream>
#include <iomanip>

namespace mypcl {

#define RESET "\033[0m"
#define BLACK "\033[30m"              /* Black */
#define RED "\033[31m"                /* Red */
#define GREEN "\033[32m"              /* Green */
#define YELLOW "\033[33m"             /* Yellow */
#define BLUE "\033[34m"               /* Blue */
#define MAGENTA "\033[35m"            /* Magenta */
#define CYAN "\033[36m"               /* Cyan */
#define WHITE "\033[37m"              /* White */
#define BOLDBLACK "\033[1m\033[30m"   /* Bold Black */
#define BOLDRED "\033[1m\033[31m"     /* Bold Red */
#define BOLDGREEN "\033[1m\033[32m"   /* Bold Green */
#define BOLDYELLOW "\033[1m\033[33m"  /* Bold Yellow */
#define BOLDBLUE "\033[1m\033[34m"    /* Bold Blue */
#define BOLDMAGENTA "\033[1m\033[35m" /* Bold Magenta */
#define BOLDCYAN "\033[1m\033[36m"    /* Bold Cyan */
#define BOLDWHITE "\033[1m\033[37m"   /* Bold White */

#define YellowCHECK(condition, _log) \
    CHECK(condition) << BOLDYELLOW << _log << RESET;

#define ColorCHECK(condition, _log, _color) \
    CHECK(condition) << _color << _log << RESET;

#define RedLOG_IF(severty, condition, _log) \
    LOG_IF(severty, condition) << BOLDRED << _log << RESET;

#define YellowLOG_IF(severty, condition, _log) \
    LOG_IF(severty, condition) << BOLDYELLOW << _log << RESET;

#define GreenLOG_IF(severty, condition, _log) \
    LOG_IF(severty, condition) << BOLDGREEN << _log << RESET;

#define ColorLOG_IF(severty, condition, _log, _color) \
    LOG_IF(severty, condition) << _color << _log << RESET;

#define ColorLOG_EVERY_N(severty, interval, _log, _color) \
    LOG_EVERY_N(severty, interval) << _color << _log << RESET;

}  // namespace mypcl
