from typing import Union

import torch
from torch.nn import functional as F

from vocalocator.outputs.base import BaseDistributionOutput, Unit


class GaussianOutput(BaseDistributionOutput):
    """
    Base class representing a (batch of) Gaussian response distribution(s) that
    packages model output together with its units.
    """

    def __init__(
        self,
        raw_output: torch.Tensor,
        arena_dims: torch.Tensor,
        arena_dim_units: Unit,
    ):
        super().__init__(raw_output, arena_dims, arena_dim_units)

        # expect subclasses to provide this instance variable
        # in their init methods
        self.cholesky_covs: torch.Tensor

        self.nnode: int  # Should be provided by subclasses
        self.ndim: int  # Should be provided by subclasses

        if raw_output.shape[-1] != self.N_OUTPUTS_EXPECTED:
            raise ValueError(
                f"Given Gaussian output class {self.__class__.__name__}, expected "
                f"output to have dimension {self.N_OUTPUTS_EXPECTED} "
                f"along final axis, but instead encountered dimension "
                f"{raw_output.shape[-1]}."
            )

    def _point_estimate(self):
        """
        Return the mean of the Gaussian(s) in the specified units.
        """
        # first two values of model output are always interpreted as the mean
        raw_output = self.raw_output
        return raw_output[:, : self.ndim * self.nnode].reshape(
            -1, self.nnode, self.ndim
        )

    def _log_p(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return log p(x) under the Gaussian parameterized by the model
        output.

        Expects x to have shape (..., self.batch_size, self.nnode, self.ndim), and to be provided
        in arbitrary units (i.e. on the square [-1, 1]^2).
        """
        # Subclasses should have shape checks to ensure the correct number of nodes
        # is provided
        x = x[..., : self.nnode, : self.ndim]  # This output only uses one node
        x = x.reshape(*x.shape[:-2], self.nnode * self.ndim)

        # The center of the distribution needs to be in (nnode*ndim,) space instead of
        # (nnode, ndim) space
        dist_center = self.point_estimate().reshape(-1, self.nnode * self.ndim)
        distr = torch.distributions.MultivariateNormal(
            loc=dist_center, scale_tril=self.cholesky_covs
        )

        return distr.log_prob(x)

    def covs(self, units: Unit) -> torch.Tensor:
        """
        Return the covariance matrix/matrices of the Gaussian(s), in
        the specified units.
        """
        # let S be a covariance matrix parameterized as S = L @ L.T
        # for L the Cholesky factor stored in self.cholesky_covs
        # then for A the linear part of the affine transformation
        # from arbitrary units to the real arena size, the resulting
        # transformed covariance matrix is given by A @ S @ A.T.
        # calculate this as the equivalent: (A @ L) @ (A @ L).T
        scaled_cholesky = self.cholesky_covs
        if units != Unit.ARBITRARY:
            scale_factor = 0.5 * self.arena_dims[units].max()
            scaled_cholesky = scale_factor * scaled_cholesky
        return scaled_cholesky @ scaled_cholesky.swapaxes(-2, -1)


class GaussianOutput3dIndependentHeight(GaussianOutput):
    N_OUTPUTS_EXPECTED = 7
    config_name = "GAUSSIAN_3D_INDEPENDENT_HEIGHT"
    computes_calibration = False

    def __init__(
        self,
        raw_output: torch.Tensor,
        arena_dims: torch.Tensor,
        arena_dim_units: Unit,
    ):
        super().__init__(raw_output, arena_dims, arena_dim_units)

        # expect subclasses to provide this instance variable
        # in their init methods
        self.cholesky_covs: torch.Tensor

        self.nnode: int = 1
        self.ndim: int = 3

        self.plane_raw_output = raw_output[
            :, : GaussianOutputFullCov.N_OUTPUTS_EXPECTED
        ]
        self.height_raw_output = raw_output[
            :, GaussianOutputFullCov.N_OUTPUTS_EXPECTED :
        ]

        self.plane_output = GaussianOutputFullCov(
            self.plane_raw_output, arena_dims, arena_dim_units
        )

    def _point_estimate(self):
        """
        Return the mean of the Gaussian(s) in the specified units.
        """
        # first two values of model output are always interpreted as the mean
        xy = self.plane_raw_output[:, :2]
        height = self.height_raw_output[:, 0]
        return torch.cat((xy, height[:, None]), dim=-1).unsqueeze(-2)

    def _log_p(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return log p(x) under the Gaussian parameterized by the model
        output.

        Expects x to have shape (..., self.batch_size, 1, 3), and to be provided
        in arbitrary units (i.e. on the cube [-1, 1]^3).
        """
        # Handle the case where there are multiple nodes in the ground truth location
        # Subclasses should override this
        if not (len(x.shape) > 2 and x.shape[-3] == self.batch_size):
            raise ValueError(
                "Incorrect shape for input! Expected last two dimensions to be "
                "(num_nodes, num_dims), but instead found shape {x.shape}."
            )

        # Handle the case where the dimensions of x are not the same as that of the
        # model output
        if x.shape[-1] < self.ndim:
            # Issue warning if ground truth location does not contain height info
            raise ValueError(
                f"Expected ground truth location to contain height information, "
                f"but instead found shape {x.shape}."
            )

        x = x[..., : self.nnode, : self.ndim].reshape(
            *x.shape[:-2], self.nnode * self.ndim
        )

        height_scale = F.softplus(self.height_raw_output[:, 1])
        distr_plane = torch.distributions.MultivariateNormal(
            loc=self.plane_output.point_estimate(),
            scale_tril=self.plane_output.cholesky_covs,
        )
        distr_height = torch.distributions.Normal(
            loc=self.height_raw_output[:, 0], scale=height_scale
        )
        return distr_plane.log_prob(x[:, :2]) + distr_height.log_prob(x[:, 2])

    def covs(self, units: Unit) -> torch.Tensor:
        covariance = torch.zeros(self.batch_size, 3, 3, device=self.device)
        covariance[:, :2, :2] = self.plane_output.covs(units)
        covariance[:, 2, 2] = (
            F.softplus(self.height_raw_output[:, 1]) * self.arena_dims[units].max() / 2
        ) ** 2
        return covariance


class GaussianOutput3dFullCov(GaussianOutput):
    N_OUTPUTS_EXPECTED = 9
    config_name = "GAUSSIAN_3D_FULL_COV"
    computes_calibration = False

    def __init__(
        self,
        raw_output: torch.Tensor,
        arena_dims: torch.Tensor,
        arena_dim_units: Unit,
    ):
        super().__init__(raw_output, arena_dims, arena_dim_units)

        self.nnode: int = 1
        self.ndim: int = 3

        L = torch.zeros(self.batch_size, 3, 3, device=self.device)
        # embed the elements into the matrix
        idxs = torch.tril_indices(3, 3)
        L[:, idxs[0], idxs[1]] = self.raw_output[:, 3:]
        # apply softplus to the diagonal entries to guarantee the resulting
        # matrix is positive definite
        new_diagonals = F.softplus(L.diagonal(dim1=-2, dim2=-1))
        L = L.diagonal_scatter(new_diagonals, dim1=-2, dim2=-1)
        # reshape y_hat so we can concatenate it to L
        self.cholesky_covs = L

    def _log_p(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return log p(x) under the Gaussian parameterized by the model
        output.

        Expects x to have shape (..., self.batch_size, 3), and to be provided
        in arbitrary units (i.e. on the square [-1, 1]^3).
        """
        # Handle the case where there are multiple nodes in the ground truth location
        # Subclasses should override this
        if not (len(x.shape) > 2 and x.shape[-3] == self.batch_size):
            raise ValueError(
                "Incorrect shape for input! Expected last two dimensions to be "
                "(num_nodes, num_dims), but instead found shape {x.shape}."
            )

        # Handle the case where the dimensions of x are not the same as that of the
        # model output
        if x.shape[-1] < self.ndim:
            # Issue warning if ground truth location does not contain height info
            raise ValueError(
                f"Expected ground truth location to contain height information, "
                f"but instead found shape {x.shape}."
            )

        x = x[..., : self.nnode, : self.ndim].reshape(
            *x.shape[:-2], self.nnode * self.ndim
        )
        distr = torch.distributions.MultivariateNormal(
            loc=self.point_estimate(), scale_tril=self.cholesky_covs
        )
        return distr.log_prob(x)

    def covs(self, units: Unit) -> torch.Tensor:
        covariance = self.cholesky_covs @ self.cholesky_covs.swapaxes(-2, -1)
        if units != Unit.ARBITRARY:
            scale = 0.5 * self.arena_dims[units].max()
            covariance = scale**2 * covariance
        return covariance


class GaussianOutput2dOriented(GaussianOutput):
    N_OUTPUTS_EXPECTED = 7
    config_name = "GAUSSIAN_2D_ORIENTED"

    def __init__(
        self,
        raw_output: torch.Tensor,
        arena_dims: torch.Tensor,
        arena_dim_units: Unit,
    ):
        """Parametrizes a distribution for both the source location and source directivity.
        First 2 components are mean location, next 3 components are the lower triangular
        Cholesky factor of the covariance matrix for the location, next 2 components are
        the mean and scale of the Von Mises distribution for the orientation.
        """
        super().__init__(raw_output, arena_dims, arena_dim_units)

        self.nnode: int = 2
        self.ndim: int = 2

        L = torch.zeros(self.batch_size, 2, 2, device=self.device)
        # embed the elements into the matrix
        idxs = torch.tril_indices(2, 2)
        L[:, idxs[0], idxs[1]] = self.raw_output[:, 2:-2]
        # apply softplus to the diagonal entries to guarantee the resulting
        # matrix is positive definite
        new_diagonals = F.softplus(L.diagonal(dim1=-2, dim2=-1))
        L = L.diagonal_scatter(new_diagonals, dim1=-2, dim2=-1)

        self.cholesky_covs = L

    def _log_p(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return log p(x) under the Gaussian parameterized by the model
        output.

        Expects x to have shape (..., self.batch_size, 2, 2), and to be provided
        in arbitrary units (i.e. on the square [-1, 1]^4).
        """
        # Handle the case where there are multiple nodes in the ground truth location
        # Subclasses should override this
        if not (len(x.shape) > 2 and x.shape[-3] == self.batch_size):
            raise ValueError(
                "Incorrect shape for input! Expected last two dimensions to be "
                "(num_nodes, num_dims), but instead found shape {x.shape}."
            )

        x = x[..., : self.nnode, : self.ndim]

        nose = x[..., 0, :]
        head = x[..., 1, :]

        orientation = nose - head
        orientation_angle = torch.atan2(orientation[..., 1], orientation[..., 0])

        distr = torch.distributions.MultivariateNormal(
            loc=self.point_estimate(), scale_tril=self.cholesky_covs
        )
        orientation_distr = torch.distributions.VonMises(
            loc=self.raw_output[:, 5], concentration=F.softplus(self.raw_output[:, 6])
        )

        return distr.log_prob(nose) + orientation_distr.log_prob(orientation_angle)

    def angle(self):
        """Mean of the Von Mises distribution over source orientation."""
        return self.raw_output[:, 5]

    def concentration(self):
        """Concentration parameter of the Von Mises distribution over source orientation."""
        return F.softplus(self.raw_output[:, 6])


class GaussianOutput3dOriented(GaussianOutput):
    N_OUTPUTS_EXPECTED = 11
    config_name = "GAUSSIAN_3D_ORIENTED"

    def __init__(
        self,
        raw_output: torch.Tensor,
        arena_dims: torch.Tensor,
        arena_dim_units: Unit,
    ):
        """Parametrizes a distribution for both the source location and source directivity.
        First 3 components are mean location, next 6 components are the lower triangular
        Cholesky factor of the covariance matrix for the location, next 2 components are
        the mean and scale of the Von Mises distribution for the orientation.

        This class is for predictions of 3d location and 2d orientation.
        """
        super().__init__(raw_output, arena_dims, arena_dim_units)

        self.nnode: int = 2
        self.ndim: int = 3

        L = torch.zeros(self.batch_size, 3, 3, device=self.device)
        # embed the elements into the matrix
        idxs = torch.tril_indices(3, 3)
        L[:, idxs[0], idxs[1]] = self.raw_output[:, 3:-2]
        # apply softplus to the diagonal entries to guarantee the resulting
        # matrix is positive definite
        new_diagonals = F.softplus(L.diagonal(dim1=-2, dim2=-1))
        L = L.diagonal_scatter(new_diagonals, dim1=-2, dim2=-1)

        self.cholesky_covs = L

    def _log_p(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return log p(x) under the Gaussian parameterized by the model
        output.

        Expects x to have shape (..., self.batch_size, 2, 2), and to be provided
        in arbitrary units (i.e. on the square [-1, 1]^4).
        """
        # Handle the case where there are multiple nodes in the ground truth location
        # Subclasses should override this
        if not (len(x.shape) > 2 and x.shape[-3] == self.batch_size):
            raise ValueError(
                "Incorrect shape for input! Expected last two dimensions to be "
                "(num_nodes, num_dims), but instead found shape {x.shape}."
            )

        x = x[..., : self.nnode, : self.ndim]

        nose = x[..., 0, :]
        head = x[..., 1, :]
        orientation = nose - head

        orientation_angle = torch.atan2(orientation[..., 1], orientation[..., 0])

        distr = torch.distributions.MultivariateNormal(
            loc=self.point_estimate(), scale_tril=self.cholesky_covs
        )
        orientation_distr = torch.distributions.VonMises(
            loc=self.raw_output[:, -2], concentration=F.softplus(self.raw_output[:, -1])
        )

        # Note that this may require a different learning rate
        return distr.log_prob(nose) + orientation_distr.log_prob(orientation_angle)

    def angle(self):
        """Mean of the Von Mises distribution over source orientation."""
        return self.raw_output[:, -2]

    def concentration(self):
        """Concentration parameter of the Von Mises distribution over source orientation."""
        return F.softplus(self.raw_output[:, -1])


class GaussianOutput4dOriented(GaussianOutput):
    N_OUTPUTS_EXPECTED = 14
    config_name = "GAUSSIAN_4D_ORIENTED"
    computes_calibration = False

    def __init__(
        self,
        raw_output: torch.Tensor,
        arena_dims: torch.Tensor,
        arena_dim_units: Unit,
    ):
        """Parametrizes a distribution for both source location and source directivity in 2D.
        First two outputs are the source location, second two are for a point behind the source,
        e.g. the head point. The final 10 outputs are the lower triangular Cholesky factor of the
        covariance matrix for the two points.
        """
        super().__init__(raw_output, arena_dims, arena_dim_units)

        self.nnode: int = 2
        self.ndim: int = 2

        L = torch.zeros(self.batch_size, 4, 4, device=self.device)
        # embed the elements into the matrix
        idxs = torch.tril_indices(4, 4)
        L[:, idxs[0], idxs[1]] = self.raw_output[:, 4:]
        # apply softplus to the diagonal entries to guarantee the resulting
        # matrix is positive definite
        new_diagonals = F.softplus(L.diagonal(dim1=-2, dim2=-1))
        L = L.diagonal_scatter(new_diagonals, dim1=-2, dim2=-1)

        self.cholesky_covs = L


class GaussianOutput6dOriented(GaussianOutput):
    N_OUTPUTS_EXPECTED = 27
    config_name = "GAUSSIAN_6D_ORIENTED"
    computes_calibration = False

    def __init__(
        self,
        raw_output: torch.Tensor,
        arena_dims: torch.Tensor,
        arena_dim_units: Unit,
    ):
        """Parametrizes a distribution for both source location and source directivity in 3D.
        First three outputs are the source location, next three are for a point behind the source,
        e.g. the head point. The final 21 outputs are the lower triangular Cholesky factor of the
        covariance matrix for the two points.
        """
        super().__init__(raw_output, arena_dims, arena_dim_units)

        self.nnode: int = 2
        self.ndim: int = 3

        L = torch.zeros(self.batch_size, 6, 6, device=self.device)
        # embed the elements into the matrix
        idxs = torch.tril_indices(6, 6)
        L[:, idxs[0], idxs[1]] = self.raw_output[:, 6:]
        # apply softplus to the diagonal entries to guarantee the resulting
        # matrix is positive definite
        new_diagonals = F.softplus(L.diagonal(dim1=-2, dim2=-1))
        L = L.diagonal_scatter(new_diagonals, dim1=-2, dim2=-1)

        self.cholesky_covs = L


class GaussianOutputFixedVariance(GaussianOutput):
    N_OUTPUTS_EXPECTED = 2
    config_name = "GAUSSIAN_FIXED_VARIANCE"

    def __init__(
        self,
        raw_output: torch.Tensor,
        arena_dims: torch.Tensor,
        arena_dim_units: Unit,
        variance: torch.Tensor,
        units: Union[str, Unit],
    ):
        """
        Construct a GaussianOutput object with fixed spherical covariance
        matrix across all inputs, with diagonal entries given by `variance` in
        units `units`.
        """
        super().__init__(raw_output, arena_dims, arena_dim_units)

        self.nnode: int = 1
        self.ndim: int = 2

        if not variance or not units:
            raise ValueError(
                "For spherical fixed variance parameterization, "
                "arguments `variance` and `units` are required!"
            )
        variance = variance.to(self.device)
        if isinstance(units, str):
            units = Unit[units]
        # if necessary, convert to arbitrary units
        if units == Unit.ARBITRARY:
            variances = torch.ones(2, device=self.device) * variance
        else:
            variances = (
                torch.ones(2, device=self.device) * variance
            ) / self.arena_dims[units].max()

        cholesky_cov = torch.diag_embed(torch.sqrt(variances))
        self.cholesky_covs = cholesky_cov.unsqueeze(0).repeat(self.batch_size, 1, 1)


class GaussianOutputSphericalCov(GaussianOutput):
    N_OUTPUTS_EXPECTED = 3
    config_name = "GAUSSIAN_SPHERICAL_COV"

    def __init__(
        self,
        raw_output: torch.Tensor,
        arena_dims: torch.Tensor,
        arena_dim_units: Unit,
    ):
        """
        Construct a GaussianOutput object with spherical covariance matrices.

        Unlike GaussianOutputFixedVariance, this class does NOT impose the
        restriction that covariance matrices are fixed, instead they may vary
        by input.
        """
        super().__init__(raw_output, arena_dims, arena_dim_units)

        self.nnode: int = 1
        self.ndim: int = 2
        # for now, just do a softplus so the diagonal entries of the
        # Cholesky (lower triangular) factor of the covariance matrix
        # are positive
        diagonal_entries = F.softplus(self.raw_output[:, 2])
        # (self.batch_size, 1, 1) * (2, 2) -> (self.batch_size, 2, 2)
        self.cholesky_covs = diagonal_entries[:, None, None] * torch.eye(
            2, device=self.device
        )


class GaussianOutputDiagonalCov(GaussianOutput):
    N_OUTPUTS_EXPECTED = 4
    config_name = "GAUSSIAN_DIAGONAL_COV"

    def __init__(
        self,
        raw_output: torch.Tensor,
        arena_dims: torch.Tensor,
        arena_dim_units: Unit,
    ):
        """
        Construct a GaussianOutput object with diagonal covariance matrices.
        """
        super().__init__(raw_output, arena_dims, arena_dim_units)

        self.nnode: int = 1
        self.ndim: int = 2

        # for now, just do a softplus so the diagonal entries of the
        # Cholesky (lower triangular) factor of the covariance matrix
        # are positive
        diagonal_entries = F.softplus(self.raw_output[:, 2:])  # (self.batch_size, 2)
        self.cholesky_covs = torch.diag_embed(
            diagonal_entries
        )  # (self.batch_size, 2, 2)


class GaussianOutputFullCov(GaussianOutput):
    N_OUTPUTS_EXPECTED = 5
    config_name = "GAUSSIAN_FULL_COV"

    def __init__(
        self,
        raw_output: torch.Tensor,
        arena_dims: torch.Tensor,
        arena_dim_units: Unit,
    ):
        """
        Construct a GaussianOutput object with diagonal covariance matrices.
        """
        super().__init__(raw_output, arena_dims, arena_dim_units)

        self.nnode: int = 1
        self.ndim: int = 2

        L = torch.zeros(self.batch_size, 2, 2, device=self.device)
        # embed the elements into the matrix
        idxs = torch.tril_indices(2, 2)
        L[:, idxs[0], idxs[1]] = self.raw_output[:, 2:]
        # apply softplus to the diagonal entries to guarantee the resulting
        # matrix is positive definite
        new_diagonals = F.softplus(L.diagonal(dim1=-2, dim2=-1))
        L = L.diagonal_scatter(new_diagonals, dim1=-2, dim2=-1)
        # reshape y_hat so we can concatenate it to L
        self.cholesky_covs = L
