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
## @deftypefn {} {@var{retval} =} magic_with_pca (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: Andrei <Andrei@DESKTOP-PK505U9>
## Created: 2021-09-08

function [train, miu, F, Vk] = magic_with_pca(train_mat, pcs)

    [m, n] = size(train_mat);

    train = zeros(m, n);

    miu = zeros(1, n);

    F = zeros(m, pcs);

    Vk = zeros(n, pcs);

    train_mat = double(train_mat);

    miu = mean(train_mat);

    centered_data = train_mat - miu;

    cov_matrix = centered_data' * centered_data / (m - 1);

    [V, D] = eig(cov_matrix);

    [~, indices] = sort(diag(D), 'descend');

    V = V(:, indices);

    Vk = V(:, 1:pcs);

    F = centered_data * Vk;

    train = F * Vk';
    
end