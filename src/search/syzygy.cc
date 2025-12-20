/*
  Seer is a UCI chess engine by Connor McMonigle
  Copyright (C) 2021-2023  Connor McMonigle

  Seer is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Seer is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <search/syzygy.h>

#include <search/atomic_tbprobe.h>

namespace search::syzygy {

tb_dtz_result tb_dtz_result::from_value(const chess::board&, const unsigned int&) noexcept { return tb_dtz_result::failure(); }

tb_wdl_result probe_wdl(const chess::board& bd) noexcept {
  atomic_tb::ProbeResult res{};
  if (!atomic_tb::probe_wdl(bd, res)) { return tb_wdl_result::failure(); }

  const wdl_type wdl = (res.wdl == atomic_tb::WDL::Win)  ? wdl_type::win
                         : (res.wdl == atomic_tb::WDL::Loss) ? wdl_type::loss
                                                             : wdl_type::draw;
  return tb_wdl_result{true, wdl};
}

tb_dtz_result probe_dtz(const chess::board& bd) noexcept {
  atomic_tb::ProbeResult res{};
  if (!atomic_tb::probe_dtz(bd, res)) { return tb_dtz_result::failure(); }

  const search::score_type score = (res.wdl == atomic_tb::WDL::Win)
                                       ? search::tb_win_score
                                       : (res.wdl == atomic_tb::WDL::Loss) ? search::tb_loss_score : search::draw_score;
  return tb_dtz_result{true, score, chess::move::null()};
}

void init(const std::string& path) noexcept { atomic_tb::init(path); }

}  // namespace search::syzygy
