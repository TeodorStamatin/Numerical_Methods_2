## Copyright (C) 2023 Andrei
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
## @deftypefn {} {@var{retval} =} task2 (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: Andrei <Andrei@DESKTOP-PK505U9>
## Created: 2023-02-28

function new_X = task2(photo, pcs)

  photo = double(photo);
  
  mu = mean(photo, 2);

  photo_centered = photo - mu;
  
  Z = transpose(photo_centered) / sqrt(size(photo_centered, 1) - 1);
  
  [U, S, V] = svd(Z);
  
  W = V(:, 1:pcs);
  
  Y = transpose(W) * photo_centered;
  
  photo_approx = W * Y + mu;
  
  new_X = uint8(photo_approx);

end