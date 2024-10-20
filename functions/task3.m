## Copyright (C) 2021 Andrei
##
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <https://www.gnu.org/licenses/>.

## -*- texinfo -*-
## @deftypefn {} {@var{retval} =} pca_cov (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: Andrei <Andrei@DESKTOP-PK505U9>
## Created: 2021-09-06

function new_X = task3(photo, pcs)

  [m, n] = size(photo);

  new_X = zeros(m, n);
  
  photo = double(photo);

  row_means = mean(photo, 2);

  photo_norm = photo - row_means;

  cov_mat = photo_norm * photo_norm' / (n - 1);

  [V, D] = eig(cov_mat);

  [D_sorted, idx] = sort(diag(D), 'descend');

  V_sorted = V(:, idx);
  
  V_pcs = V_sorted(:, 1:pcs);

  Y = V_pcs' * photo_norm;

  new_X_norm = V_pcs * Y;

  new_X = new_X_norm + row_means;

  new_X = uint8(new_X);

  end
